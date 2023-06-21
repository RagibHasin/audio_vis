#![allow(clippy::excessive_precision)]

use std::{f32::consts, ops::RangeInclusive};

use anyhow::Context;
use euclid::{
    default::{Point2D, Size2D, Vector2D},
    size2, vec2,
};
use serde::Deserialize;

pub type BufIter = Box<dyn Iterator<Item = ((i16, i16), (i16, i16))>>;

pub const STEP: usize = 1200;
pub const F_MAX: f32 = 20e3;
pub const T_SWEET: f32 = 4.32;
pub const GOLDEN_RATIO: f32 = 0.6085538238;

fn gamma_c(x: f32) -> f32 {
    const A0: f32 = 0.25;
    const A1: f32 = 0.225432981868225421;
    const A2: f32 = 9.57111578549689866;
    const A3: f32 = 0.463622652910641416;
    A0 + A1 * (A2 * x + A3).log2()
}

fn digamma_u(y: f32) -> f32 {
    const A0: f32 = 0.10448102628904552;
    const A1: f32 = 4.4359081431329331;
    const A2: f32 = 0.173287;
    const A3: f32 = 0.69314718055994531;
    const A4: f32 = 0.46362265291064142;
    A0 * ((A1 * (-A2 + A3 * y)).exp() - A4)
}

pub struct Model {
    pub sample_rate: u32,
    pub audio_iter: BufIter,
    pub audio_buffer: [Vec<(i16, i16)>; 2],
    pub sample_buffer: [Vec<PartialSampleDesc>; 2],
    pub k_offset: usize,
    pub config: Config,
}

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(deserialize_with = "de_size")]
    pub viewport_size: Size2D<u32>,
    pub vol_per_half_turn: i16,
    pub linear_position: bool,
    pub independent_luma: bool,
    pub luma_range: RangeInclusive<f32>,
    pub chroma_range: RangeInclusive<f32>,
    pub hue_range: RangeInclusive<f32>,
    pub hue_offset: f32,
    pub fuzz_factor: f32,
}

fn de_size<'de, D>(deserializer: D) -> Result<Size2D<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    parse_size(&String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}

pub fn parse_size(s: &str) -> anyhow::Result<Size2D<u32>> {
    let mut s = s.split('x');
    Ok(size2(
        s.next().context("invalid viewport size")?.parse()?,
        s.next().context("invalid viewport size")?.parse()?,
    ))
}

impl Config {
    fn rad_per_vol(&self) -> f32 {
        consts::PI / self.vol_per_half_turn as f32
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            viewport_size: size2(960, 540),
            vol_per_half_turn: 5120,
            linear_position: false,
            independent_luma: false,
            luma_range: 0.0..=1.,
            chroma_range: 0.0..=2.8,
            hue_range: 0.0..=360.,
            hue_offset: 0.,
            fuzz_factor: 0.2,
        }
    }
}

impl Model {
    pub fn buffer_len(&self) -> usize {
        self.audio_buffer[0].len()
    }

    pub fn expented_buffer_len(&self) -> usize {
        (self.sample_rate as f32 * T_SWEET) as usize / STEP + 1
    }

    pub fn new(sample_rate: u32, audio_iter: BufIter, config: Config) -> anyhow::Result<Model> {
        let buffer_size = (sample_rate as f32 * T_SWEET) as usize / STEP;
        let audio_buffer = [vec![(0, 0); buffer_size], vec![(0, 0); buffer_size]];
        let sample_buffer = [
            vec![PartialSampleDesc::default(); buffer_size],
            vec![PartialSampleDesc::default(); buffer_size],
        ];

        tracing::info!(?config);

        Ok(Model {
            sample_rate,
            audio_iter,
            audio_buffer,
            sample_buffer,
            k_offset: 0,
            config,
        })
    }

    pub fn update(&mut self, since_last: std::time::Duration) -> bool {
        let elapsed_audio_sample = (since_last.as_nanos() * self.sample_rate as u128
            / 1_000_000_000) as usize
            + self.k_offset;
        let take = elapsed_audio_sample / STEP;
        self.k_offset = elapsed_audio_sample % STEP;

        let buffer_len = self.buffer_len();
        for buf in &mut self.audio_buffer {
            if take < buffer_len {
                buf.drain(0..take);
            } else {
                buf.clear();
            }
        }
        for buf in &mut self.sample_buffer {
            if take < buffer_len - 1 {
                buf.drain(0..take);
            } else {
                buf.clear();
            }
        }

        for ((last_channel_0, last_channel_1), (now_channel_0, now_channel_1)) in self
            .audio_iter
            .by_ref()
            .skip(take.saturating_sub(buffer_len))
            .take(take.min(buffer_len))
        {
            self.audio_buffer[0].push((last_channel_0, now_channel_0));
            self.audio_buffer[1].push((last_channel_1, now_channel_1));
        }

        let lin_freq_coeff = self.sample_rate as f32 / consts::TAU / F_MAX;
        for (audio_buffer, sample_buffer) in self.audio_buffer.iter().zip(&mut self.sample_buffer) {
            sample_buffer.extend(audio_buffer.iter().copied().skip(buffer_len - take).scan(
                sample_buffer.last().map_or(GOLDEN_RATIO, |c| c.position),
                |prev_position, (audio_, audio)| {
                    let config = &self.config;
                    let phase = audio as f32 * config.rad_per_vol();
                    let velocity = (audio as i32 - audio_ as i32) as f32;
                    let frac_freq = (velocity * config.rad_per_vol() * lin_freq_coeff) % 1.;

                    let position = *prev_position
                        + if velocity <= 0. {
                            *prev_position
                        } else {
                            1. - *prev_position
                        } * if config.linear_position {
                            velocity * GOLDEN_RATIO / i16::MAX as f32
                        } else {
                            gamma_c((velocity / i16::MAX as f32).clamp(-1., 1.).abs())
                                .copysign(velocity)
                        };

                    let audio = to_f32(audio);
                    let chroma = (audio / phase.cos())
                        .abs()
                        .clamp(*config.chroma_range.start(), *config.chroma_range.end());
                    let hue = (phase.rem_euclid(consts::TAU) / consts::TAU
                        * (config.hue_range.end() - config.hue_range.start())
                        + *config.hue_range.start()
                        + config.hue_offset)
                        .to_radians();

                    *prev_position = position;

                    Some(PartialSampleDesc {
                        audio,
                        frac_freq,
                        chroma,
                        hue,
                        position,
                    })
                },
            ));
        }

        self.buffer_len() != self.expented_buffer_len() // is finished?
    }

    pub fn render(&self) -> impl Iterator<Item = SampleDesc> + '_ {
        let vp = self.config.viewport_size.cast::<f32>();
        let aspect_ratio = vp.width / vp.height;

        let buffer_size = self.buffer_len();

        let sample_filler = move |(
            j,
            (
                i,
                &PartialSampleDesc {
                    audio,
                    frac_freq,
                    chroma,
                    hue,
                    position,
                },
            ),
        )| {
            let config = &self.config;
            let channel = j % 2;
            let k = (buffer_size - i - 2) * STEP + self.k_offset;

            let abs_frac_freq = frac_freq.rem_euclid(1.);
            let decay_time = T_SWEET * 2f32.powf(-abs_frac_freq);

            let spreadth = 0.5 * k as f32 / decay_time / self.sample_rate as f32;
            if spreadth >= 1. {
                return None;
            }
            let spreadth = if frac_freq < 0. {
                1. - spreadth
            } else {
                spreadth
            };

            let radius = vp.width / 2. * spreadth;
            let reduced_vp: Size2D<_> = size2(aspect_ratio, 1.) * 1080.;
            let channel_area =
                0.5 * reduced_vp.height * (2. * reduced_vp.width - reduced_vp.height - 1.);
            let cool_area = reduced_vp.height * (reduced_vp.width - reduced_vp.height);

            let position = position * channel_area;

            let (x, y) = if position < cool_area {
                let x_ = position.div_euclid(reduced_vp.height);
                let y = if x_.rem_euclid(2.) == 0. {
                    position.rem_euclid(reduced_vp.height)
                } else {
                    reduced_vp.height - position.rem_euclid(reduced_vp.height)
                };
                (x_ + y, y)
            } else {
                let x_ = reduced_vp.width - reduced_vp.height;
                let uncool_position = position - cool_area;
                if x_.rem_euclid(2.) == 0. {
                    // 0.5 x (x+1) = position - cool_area
                    // => x^2 + x - 2 (position - cool_area) = 0
                    // => x = 0.5 (-1 + sqrt(1 + 8 (position - cool_area)))
                    let x = (0.5 * ((1. + 8. * uncool_position).sqrt() - 1.)).floor();
                    let y = uncool_position - 0.5 * x * (x + 1.);
                    (x_ + x, y)
                } else {
                    // channel_area - 0.5 x (x+1) = position - cool_area
                    // 0.5 x (x+1) = channel_area - (position - cool_area)
                    // => x^2 + x - 2 (channel_area - (position - cool_area)) = 0
                    // => x = 0.5 (-1 + sqrt(1 + 8 (channel_area - (position - cool_area))))
                    let x =
                        (0.5 * ((1. + 8. * (channel_area - uncool_position)).sqrt() - 1.)).ceil();
                    let y = channel_area - uncool_position - 0.5 * x * (x + 1.);
                    (reduced_vp.width - x - y, reduced_vp.height - y)
                }
            };
            let position = vec2(x, y) / 1080.;
            let center = (if channel == 1 {
                vec2(aspect_ratio, 1.) - position
            } else {
                position
            } * vp.height)
                .to_point();

            let luma = if self.config.independent_luma {
                gamma_c(abs_frac_freq)
            } else {
                (gamma_c(abs_frac_freq) * gamma_c(audio.abs())).powf(GOLDEN_RATIO)
            };
            let luma = luma * (config.luma_range.end() - config.luma_range.start())
                + *config.luma_range.start();
            let (a, b) =
                Vector2D::from_angle_and_length(euclid::Angle { radians: hue }, chroma).to_tuple();
            let alpha = 1. - gamma_c(spreadth);
            let fuzz = digamma_u(spreadth);

            Some(SampleDesc {
                luma,
                a,
                b,
                alpha,
                center,
                radius_in: radius * (1. - self.config.fuzz_factor * fuzz),
                radius_out: radius * (1. + self.config.fuzz_factor * fuzz),
            })
        };

        itertools::interleave(
            self.sample_buffer[0].iter().enumerate(),
            self.sample_buffer[1].iter().enumerate(),
        )
        .enumerate()
        .flat_map(sample_filler)
    }
}

fn to_f32(i: i16) -> f32 {
    i as f32 / i16::MAX as f32
}

#[derive(Debug, Clone, Copy)]
pub struct PartialSampleDesc {
    pub audio: f32,
    pub frac_freq: f32,
    pub chroma: f32,
    pub hue: f32,
    pub position: f32,
}

impl Default for PartialSampleDesc {
    fn default() -> Self {
        Self {
            audio: 0.,
            frac_freq: 0.,
            chroma: 0.,
            hue: 0.,
            position: GOLDEN_RATIO,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SampleDesc {
    pub luma: f32,
    pub a: f32,
    pub b: f32,
    pub alpha: f32,
    pub center: Point2D<f32>,
    pub radius_in: f32,
    pub radius_out: f32,
}

impl Default for SampleDesc {
    fn default() -> Self {
        Self {
            luma: 0.,
            a: 0.,
            b: 0.,
            alpha: 1.,
            center: Default::default(),
            radius_in: 0.,
            radius_out: 0.,
        }
    }
}

impl std::fmt::Debug for SampleDesc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let SampleDesc {
            luma,
            a,
            b,
            alpha,
            center,
            radius_in: radius,
            radius_out: fuzz,
        } = self;
        write!(
            f,
            "{luma:>6.4}\t{a:>6.4}\t{b:>6.4}\t{alpha:>6.4}\t({:>6.4},\t{:>6.4})\t{radius:>6.4}\t{fuzz:>1.0}",
            center.x, center.y
        )
    }
}
