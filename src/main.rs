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
    audio_path: PathBuf,
    #[arg(global = true, short = 's', long = "step", default_value_t = 1200)]
    step: usize,
    #[arg(global = true, short, long)]
    downsample: bool,
    #[arg(global = true, short = '0', long)]
    dont_prepend_zero: bool,

    #[arg(global = true, long = "config")]
    config_path: Option<PathBuf>,
    #[command(flatten)]
    config: Config,
}

#[derive(Args)]
struct Config {
    #[arg(global = true, long = "vp", value_parser = audio_vis::parse_size)]
    pub viewport_size: Option<Size2D<u32>>,
    #[arg(global = true, long = "vpht")]
    pub vol_per_half_turn: Option<i16>,
    #[arg(global = true, short, long = "linear-pos")]
    pub linear_position: bool,
    #[arg(global = true, short = 'i', long)]
    pub independent_luma: bool,
    #[arg(global = true, short = 'C', long)]
    pub linear_chroma: bool,
    #[arg(global = true, long, value_parser = parse_range)]
    pub luma_range: Option<RangeInclusive<f32>>,
    #[arg(global = true, short = 'c', long, value_parser = parse_range)]
    pub chroma_range: Option<RangeInclusive<f32>>,
    #[arg(global = true, long, value_parser = parse_range)]
    pub hue_range: Option<RangeInclusive<f32>>,
    #[arg(global = true, short = 'H', long)]
    pub hue_offset: Option<f32>,
    #[arg(global = true, short, long)]
    pub fuzz_factor: Option<f32>,
}

fn parse_range(s: &str) -> anyhow::Result<RangeInclusive<f32>> {
    let mut s = s.split(':');
    Ok(s.next().context("invalid range")?.parse()?..=s.next().context("invalid range")?.parse()?)
}

#[derive(Subcommand)]
enum Commands {
    Render {
        #[arg(short = 'x', long, default_value_t = 0)]
        skip: usize,
        #[arg(short = 'z', long, default_value_t = usize::MAX)]
        upto: usize,
        #[arg(short = 'D', long, default_value = "renders")]
        save_in: PathBuf,
    },
    View {
        #[arg(short = 'a', long)]
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
        .chain(iter::repeat((0, 0)).take((T_SWEET * sample_rate as f32) as usize));
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
