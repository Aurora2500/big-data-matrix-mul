use std::sync::Arc;
use wgpu::util::DeviceExt;

fn mat_layout(dev: &wgpu::Device) -> wgpu::BindGroupLayout {
	dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
		label: Some("matrix bind group layout"),
		entries: &[
			wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			},
			wgpu::BindGroupLayoutEntry {
				binding: 1,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Storage { read_only: false },
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			},
		],
	})
}

pub struct GPU {
	instance: wgpu::Instance,
	adapter: wgpu::Adapter,
	device: wgpu::Device,
	queue: wgpu::Queue,
	shader: wgpu::ShaderModule,
	matrix_layout: wgpu::BindGroupLayout,
}

impl GPU {
	pub async fn new() -> GPU {
		let instance = wgpu::Instance::new(Default::default());
		let adapter = instance.request_adapter(&Default::default()).await.unwrap();
		let features = adapter.features();

		let (device, queue) = adapter
			.request_device(
				&wgpu::DeviceDescriptor {
					label: None,
					features,
					limits: Default::default(),
				},
				None,
			)
			.await
			.unwrap();
		let matrix_layout = mat_layout(&device);
		let query_set = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
			Some(device.create_query_set(&wgpu::QuerySetDescriptor {
				label: Some("timestamp query set"),
				count: 2,
				ty: wgpu::QueryType::Timestamp,
			}))
		} else {
			None
		};

		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("matrix multiplication shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
		});

		return GPU {
			instance,
			adapter,
			device,
			queue,
			shader,
			matrix_layout,
		};
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Dirty {
	/// The CPU and GPU buffers are in sync.
	Clean,
	/// the GPU buffer has changes not reflected in the CPU buffer.
	CPUDirty,
	/// the CPU buffer has changes not reflected in the GPU buffer.
	GPUDirty,
}

struct MatrixGPUFrame {
	buffer: wgpu::Buffer,
	size_buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
	dirty: Dirty,
}

pub struct Matrix {
	size: usize,
	data: Vec<f32>,
	gpu: Arc<GPU>,
	gpu_frame: Option<MatrixGPUFrame>,
}

impl Matrix {
	pub fn to_gpu(&mut self, gpu: &GPU) {
		match &self.gpu_frame {
			Some(frame) => {
				// the GPU buffer is allocated, so we just need to make sure it's in sync with the CPU buffer.
				if frame.dirty == Dirty::Clean || frame.dirty == Dirty::CPUDirty {
					// The CPU doesn't have any changes, so we don't need to do anything.
					return;
				}
			}
			None => {
				// The GPU buffer isn't allocated, so we need to allocate it.
				let buffer =
					gpu.device
						.create_buffer_init(&wgpu::util::BufferInitDescriptor {
							label: Some("matrix buffer"),
							contents: bytemuck::cast_slice(&self.data),
							usage: wgpu::BufferUsages::STORAGE
								| wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
						});
				let size_buffer =
					gpu.device
						.create_buffer_init(&wgpu::util::BufferInitDescriptor {
							label: Some("matrix size buffer"),
							contents: bytemuck::cast_slice(&[self.size as u32]),
							usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
						});
				let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
					label: Some("matrix bind group"),
					layout: &gpu.matrix_layout,
					entries: &[
						wgpu::BindGroupEntry {
							binding: 0,
							resource: size_buffer.as_entire_binding(),
						},
						wgpu::BindGroupEntry {
							binding: 1,
							resource: buffer.as_entire_binding(),
						},
					],
				});
				self.gpu_frame = Some(MatrixGPUFrame {
					buffer,
					size_buffer,
					bind_group,
					dirty: Dirty::Clean,
				});
			}
		}
	}
}
