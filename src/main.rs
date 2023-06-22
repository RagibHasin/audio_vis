use std::{iter, ops::RangeInclusive, path::PathBuf};

use anyhow::Context;
use audio_vis::{BufIter, Model, T_SWEET};
use clap::{Args, Parser, Subcommand};
use euclid::default::Size2D;
use itertools::Itertools;
use pollster::block_on;
use rodio::Source;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(global = true, required = false)]
    /// Path to the audio file
    audio_path: PathBuf,
    
    #[arg(global = true, short = 's', long = "step", default_value_t = 1200)]
    /// Step size
    step: usize,
    
    #[arg(global = true, short, long)]
    /// Downsample when stepping
    downsample: bool,
    
    #[arg(global = true, short = '0', long)]
    /// Prepend zero on initialization
    dont_prepend_zero: bool,

    #[arg(global = true, long = "config")]
    /// Path to configuration file
    config_path: Option<PathBuf>,
    
    #[command(flatten)]
    config: Config,
}

#[derive(Args)]
struct Config {
    #[arg(global = true, long = "vp", value_parser = audio_vis::parse_size)]
    /// Size of the viewport
    pub viewport_size: Option<Size2D<u32>>,
    
    #[arg(global = true, long = "vpht")]
    /// Volume per half turn in audio phase
    pub vol_per_half_turn: Option<i16>,
    
    #[arg(global = true, short, long = "linear-pos")]
    /// Calculate position linearly
    pub linear_position: bool,
    
    #[arg(global = true, short = 'i', long)]
    /// Luminance independent of volume
    pub independent_luma: bool,
    
    #[arg(global = true, short = 'C', long)]
    /// Calculate chromaticity linearly
    pub linear_chroma: bool,
    
    #[arg(global = true, long, value_parser = parse_range)]
    /// Mapped range of luminance
    pub luma_range: Option<RangeInclusive<f32>>,
    
    #[arg(global = true, short = 'c', long, value_parser = parse_range)]
    /// Clipped range of chromaticity
    pub chroma_range: Option<RangeInclusive<f32>>,
    
    #[arg(global = true, long, value_parser = parse_range)]
    /// Mapped range of hue
    pub hue_range: Option<RangeInclusive<f32>>,
    
    #[arg(global = true, short = 'H', long)]
    /// Hue offset, applied before mapping
    pub hue_offset: Option<f32>,
    
    #[arg(global = true, short, long)]
    /// Factor of circle fuzziness
    pub fuzz_factor: Option<f32>,
}

fn parse_range(s: &str) -> anyhow::Result<RangeInclusive<f32>> {
    let mut s = s.split(':');
    Ok(s.next().context("invalid range")?.parse()?..=s.next().context("invalid range")?.parse()?)
}

#[derive(Subcommand)]
enum Commands {
    /// Render images at 30fps as PNG
    Render {
        #[arg(short = 'x', long, default_value_t = 0)]
        /// Number of frames to be skipped
        skip: usize,
        
        #[arg(short = 'z', long, default_value_t = usize::MAX)]
        /// Number of frame, upto which would be rendered
        upto: usize,
        
        #[arg(short = 'D', long, default_value = "renders")]
        /// Directory for the saved images
        save_in: PathBuf,
    },
    /// View visualization in real-time with optional audio
    View {
        #[arg(short = 'a', long)]
        /// Enable audio
        enable_audio: bool,
    },
}

mod cmd {
    pub mod render;
    pub mod view;
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let Cli {
        command,
        audio_path,
        step,
        downsample,
        dont_prepend_zero,
        config_path,
        config: cli_config,
    } = Cli::parse();

    let mut config: audio_vis::Config = config_path
        .map(std::fs::read)
        .transpose()?
        .as_deref()
        .map(toml_edit::de::from_slice)
        .transpose()?
        .unwrap_or_default();

    fn try_replace<T>(dest: &mut T, src: Option<T>) {
        if let Some(v) = src {
            *dest = v;
        }
    }

    try_replace(&mut config.viewport_size, cli_config.viewport_size);
    try_replace(&mut config.vol_per_half_turn, cli_config.vol_per_half_turn);
    config.linear_position |= cli_config.linear_position;
    config.independent_luma |= cli_config.independent_luma;
    config.linear_chroma |= cli_config.linear_chroma;
    try_replace(&mut config.luma_range, cli_config.luma_range);
    try_replace(&mut config.chroma_range, cli_config.chroma_range);
    try_replace(&mut config.hue_range, cli_config.hue_range);
    try_replace(&mut config.hue_offset, cli_config.hue_offset);
    try_replace(&mut config.fuzz_factor, cli_config.fuzz_factor);

    let mut audio_reader =
        rodio::Decoder::new(std::io::BufReader::new(std::fs::File::open(&audio_path)?))?;

    let sample_rate = audio_reader.sample_rate();
    let channels = audio_reader.channels();

    let audio_iter = (!dont_prepend_zero)
        .then_some((0, 0))
        .into_iter()
        .chain(
            iter::repeat_with(move || {
                let v = audio_reader.next()?;
                if channels == 1 {
                    Some((v, v))
                } else {
                    let w = audio_reader.next()?;
                    audio_reader.by_ref().dropping(channels as usize - 2);
                    Some((v, w))
                }
            })
            .flatten()
            .fuse(),
        ) // (channel_0, channel_1)
        .chain(iter::repeat((0, 0)).take((T_SWEET * sample_rate as f32).ceil() as usize));
    // ((last_channel_0, last_channel_1), (now_channel_0, now_channel_1))
    let audio_iter = if downsample {
        Box::new(audio_iter.step_by(step).tuple_windows()) as BufIter
    } else {
        Box::new(audio_iter.tuple_windows().step_by(step)) as BufIter
    };

    let model = Model::new(audio_iter, sample_rate, step, config)?;

    match command {
        Commands::Render {
            skip,
            upto,
            save_in,
        } => block_on(cmd::render::run(model, skip, upto, save_in)),
        Commands::View { enable_audio } => {
            block_on(cmd::view::run(model, audio_path, enable_audio))
        }
    }
}
