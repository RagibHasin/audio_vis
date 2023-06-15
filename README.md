# Audio visualization

For design details see [here](docs/DESIGN.pdf).

## Usage

### To view

```sh
cargo r --view -- <path_to_audio> [-a]
```

Option `-a` is for playing the audio.

### To render images

```sh
cargo r --render -- <path_to_audio> [<skipped_frames> [<taken_frames>]]
```

Rendered images are put in `renders` subdirectory of the current working directory.
The `renders` subdirectory must be present, otherwise program would panic.

## License

> The Artistic License 2.0
