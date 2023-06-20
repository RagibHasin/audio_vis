use std::{path::PathBuf, time::Duration};

use audio_vis::{Model, SampleDesc};

use euclid::size2;
use wgpu::util::DeviceExt;

struct State {
    device: wgpu::Device,
    queue: wgpu::Queue,

    texture_size: wgpu::Extent3d,
    render_out_texture: wgpu::Texture,
    output_buffer: wgpu::Buffer,

    sample_cmds: Vec<SampleDesc>,
    sample_buffer: wgpu::Buffer,
    sample_bind_group: wgpu::BindGroup,

    render_pipeline: wgpu::RenderPipeline,
    model: Model,
}

impl State {
    async fn new(model: Model) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .unwrap();

        let vp = model.config.viewport_size;
        let texture_size = wgpu::Extent3d {
            width: vp.width,
            height: vp.height,
            depth_or_array_layers: 1,
        };
        let render_out_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        });

        let output_buffer_size =
            (std::mem::size_of::<u32>() as u32 * vp.width * vp.height) as wgpu::BufferAddress;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        });

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("../shader/vertex_fragment.wgsl"));

        let sample_cmds = vec![SampleDesc::default(); model.expented_buffer_len() * 2];
        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("samples Buffer"),
            contents: bytemuck::cast_slice(&sample_cmds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("samples_bind_group_layout"),
            });

        let sample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &sample_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sample_buffer.as_entire_binding(),
            }],
            label: Some("cmd_bind_group"),
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&sample_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        State {
            device,
            queue,
            texture_size,
            render_out_texture,
            output_buffer,
            sample_cmds,
            sample_buffer,
            sample_bind_group,
            render_pipeline,
            model,
        }
    }

    fn update(&mut self, since_last: Duration) -> bool {
        self.model.update(since_last)
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.sample_cmds.clear();
        self.sample_cmds.extend(self.model.render());
        if self.sample_cmds.is_empty() {
            self.sample_cmds.push(SampleDesc::default());
        }
        self.queue.write_buffer(
            &self.sample_buffer,
            0,
            bytemuck::cast_slice(&self.sample_cmds),
        );

        let texture_view = self.render_out_texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.sample_bind_group, &[]);
        render_pass.draw(0..6, 0..1);
        drop(render_pass);

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.render_out_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(
                        std::mem::size_of::<u32>() as u32 * self.model.config.viewport_size.width,
                    ),
                    rows_per_image: Some(self.model.config.viewport_size.height),
                },
            },
            self.texture_size,
        );

        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    async fn process_image_buffer(
        &self,
        mut f: impl FnMut(image::ImageBuffer<image::Rgba<u8>, wgpu::BufferView<'_>>),
    ) {
        {
            let buffer_slice = self.output_buffer.slice(..);

            let (tx, rx) = futures_channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.await.unwrap().unwrap();

            let data = buffer_slice.get_mapped_range();

            let buffer = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                self.model.config.viewport_size.width,
                self.model.config.viewport_size.height,
                data,
            )
            .unwrap();
            f(buffer);
        }
        self.output_buffer.unmap();
    }
}

async fn run() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = pico_args::Arguments::from_env();
    let (model, _) = Model::from_args(&mut args, size2(1920, 1080))?;

    let skip = args
        .opt_value_from_fn(["-x", "--skip"], str::parse)?
        .unwrap_or(0usize);
    let upto = args
        .opt_value_from_fn(["-z", "--take"], str::parse)?
        .unwrap_or(usize::MAX);
    let save_in = args
        .opt_value_from_fn("--in", |p| {
            Ok::<_, std::convert::Infallible>(PathBuf::from(p))
        })?
        .unwrap_or_else(|| PathBuf::from("renders"));

    let mut state = State::new(model).await;

    let mut break_in_next = false;

    for frame in 0..skip {
        state.update(Duration::new(0, 33333333 + (frame % 3 == 0) as u32));
    }
    for frame in skip..upto {
        let mut rendered_path = save_in.clone();
        rendered_path.push(format!("{frame:07}.png"));
        state.render()?;
        state
            .process_image_buffer(|img| {
                img.save(&rendered_path).unwrap();
                tracing::info!("Saved frame {frame}");
            })
            .await;
        if break_in_next {
            break;
        }

        break_in_next = state.update(Duration::new(0, 33333333 + (frame % 3 == 0) as u32));
    }

    Ok(())
}

fn main() {
    pollster::block_on(run()).unwrap();
}
