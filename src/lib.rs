#![allow(clippy::excessive_precision)]

use std::{f32::consts, iter};

use euclid::{
    default::{Point2D, Size2D, Vector2D},
    point2,
};
use itertools::Itertools;
use rodio::Source;

type BufIter = Box<dyn Iterator<Item = ((i16, i16), (i16, i16))>>;

const STEP: usize = 1200;
const F_MAX: f32 = 20e3;
const T_SWEET: f32 = 4.32;
const GOLDEN_RATIO: f32 = 0.6085538238;
const VOL_PER_HALF_TURN: i16 = 5120;
const RAD_PER_VOL: f32 = consts::PI / VOL_PER_HALF_TURN as f32;

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
    pub n_samples: u64,
    pub sample_rate: u32,
    pub audio_iter: BufIter,
    pub audio_buffer: [Vec<(i16, i16)>; 2],
    pub sample_buffer: [Vec<PartialSampleDesc>; 2],
    pub k_offset: usize,
}

impl Model {
    pub fn buffer_len(&self) -> usize {
        self.audio_buffer[0].len()
    }

    pub fn expented_buffer_len(&self) -> usize {
        (self.sample_rate as f32 * T_SWEET) as usize / STEP + 1
    }

    pub fn new(path: impl AsRef<std::path::Path>) -> anyhow::Result<Model> {
        let mut audio_reader =
            rodio::Decoder::new(std::io::BufReader::new(std::fs::File::open(path.as_ref())?))?;

        let sample_rate = audio_reader.sample_rate();
        let n_samples = (audio_reader.total_duration().map_or(0, |t| t.as_nanos())
            * sample_rate as u128) as u64;
        let buffer_size = (sample_rate as f32 * T_SWEET) as usize / STEP;

        let channels = audio_reader.channels();

        let audio_iter = Box::new(
            iter::once((0, 0))
                .chain(
                    iter::repeat_with(move || {
                        let v = audio_reader.next()?;
                        if channels == 1 {
                            Some((v, v))
                        } else {
                            let w = audio_reader.next()?;
                            audio_reader
                                .by_ref()
                                .take(channels as usize - 2)
                                .for_each(|_| {});
                            Some((v, w))
                        }
                    })
                    .flatten()
                    .fuse(),
                ) // (channel_0, channel_1)
                .chain(iter::repeat((0, 0)).take((T_SWEET * 96e3) as usize))
                .tuple_windows() // ((last_channel_0, last_channel_1), (now_channel_0, now_channel_1))
                .step_by(STEP),
        );
        let audio_buffer = [vec![(0, 0); buffer_size + 1], vec![(0, 0); buffer_size + 1]];
        let sample_buffer = [
            vec![PartialSampleDesc::default(); buffer_size],
            vec![PartialSampleDesc::default(); buffer_size],
        ];

        Ok(Model {
            n_samples,
            sample_rate,
            audio_iter,
            audio_buffer,
            sample_buffer,
            k_offset: 0,
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
            sample_buffer.extend(
                audio_buffer
                    .iter()
                    .copied()
                    .skip(buffer_len - 1 - take)
                    .scan(
                        sample_buffer.last().map_or(GOLDEN_RATIO, |c| c.position),
                        |prev_position, (audio_, audio)| {
                            let phase = audio as f32 * RAD_PER_VOL;
                            let velocity = (audio as i32 - audio_ as i32) as f32;
                            let frac_freq = (velocity * RAD_PER_VOL * lin_freq_coeff) % 1.;

                            let position =
                                gamma_c((velocity / i16::MAX as f32).clamp(-1., 1.).abs())
                                    .copysign(velocity)
                                    * if velocity <= 0. {
                                        *prev_position
                                    } else {
                                        1. - *prev_position
                                    }
                                    + *prev_position;

                            let audio = to_f32(audio);
                            let chroma = (audio / phase.cos()).abs().min(2.8);
                            let hue = phase;

                            *prev_position = position;

                            Some(PartialSampleDesc {
                                audio,
                                frac_freq,
                                chroma,
                                hue,
                                position,
                            })
                        },
                    ),
            );
        }

        self.buffer_len() != self.expented_buffer_len() // is finished?
    }

    pub fn render(&self, vp: Size2D<f32>) -> impl Iterator<Item = SampleDesc> + '_ {
        let buffer_size = self.buffer_len();

        let channel_area = 0.5 * vp.height * (2. * vp.width - vp.height - 1.);
        let cool_area = vp.height * (vp.width - vp.height);

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
            let position = position * channel_area;
            let (x, y) = if position < cool_area {
                let x_ = position.div_euclid(vp.height);
                let y = if x_.rem_euclid(2.) == 0. {
                    position.rem_euclid(vp.height)
                } else {
                    vp.height - position.rem_euclid(vp.height)
                };
                (x_ + y, y)
            } else {
                let x_ = vp.width - vp.height;
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
                    (vp.width - x - y, vp.height - y)
                }
            };
            let (x, y) = if channel == 1 {
                (vp.width - x - 1., vp.height - y - 1.)
            } else {
                (x, y)
            };

            let luma = (gamma_c(abs_frac_freq) * gamma_c(audio.abs())).powf(GOLDEN_RATIO);
            let (a, b) =
                Vector2D::from_angle_and_length(euclid::Angle { radians: hue }, chroma).to_tuple();
            let alpha = 1. - gamma_c(spreadth);
            let fuzz = digamma_u(spreadth);

            Some(SampleDesc {
                radius,
                center: point2(x, y),
                luma,
                a,
                b,
                alpha,
                fuzz,
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
    pub radius: f32,
    pub fuzz: f32,
}

impl Default for SampleDesc {
    fn default() -> Self {
        Self {
            luma: 0.,
            a: 0.,
            b: 0.,
            alpha: 1.,
            center: Default::default(),
            radius: 0.,
            fuzz: 1.,
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
            radius,
            fuzz,
        } = self;
        write!(
            f,
            "{luma:>6.4}\t{a:>6.4}\t{b:>6.4}\t{alpha:>6.4}\t({:>6.4},\t{:>6.4})\t{radius:>6.4}\t{fuzz:>1.0}",
            center.x, center.y
        )
    }
}
