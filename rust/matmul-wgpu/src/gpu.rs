use wgpu::util::DeviceExt;

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

		let matrix_layout = Matrix::layout(gpu);

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
		};
	}
}

struct MatrixGPUFrame {
	buffer: wgpu::Buffer,
	size_buffer: wgpu::Buffer,
	bind_group: wgpu::BindGroup,
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

pub struct Matrix {
	size: usize,
	data: Vec<f32>,
	dirty: Dirty,
	gpu_buffer: Option<MatrixGPUFrame>,
}

impl Matrix {
	pub fn to_gpu(&mut self, gpu: &GPU) {
		if self.gpu_buffer.is_some() {
			return;
		}
		let buf = gpu
			.device
			.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("matrix buffer"),
				contents: bytemuck::cast_slice(&self.data),
				usage: wgpu::BufferUsages::STORAGE,
			});

		gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some(""),
			entries: &[
				wgpu::BindGroupEntry {
					binding: 0,
					resource: self.size,
				},
				wgpu::BindGroupEntry {
					binding: 1,
					resource: buf.as_entire_binding(),
				},
			],
			layout: &Matrix::layout(&gpu.device),
		});
	}

	fn layout(dev: &wgpu::Device) -> wgpu::BindGroupLayout {
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
}
