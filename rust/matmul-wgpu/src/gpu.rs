use std::{
	fmt::{Debug, Display},
	ops::Deref,
	sync::Arc,
};
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

pub struct Gpu {
	instance: wgpu::Instance,
	adapter: wgpu::Adapter,
	device: wgpu::Device,
	queue: wgpu::Queue,
	shader: wgpu::ShaderModule,
	pipeline_layout: wgpu::PipelineLayout,
	pipeline: wgpu::ComputePipeline,
	matrix_layout: wgpu::BindGroupLayout,
}

impl Gpu {
	pub async fn new() -> Gpu {
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
		let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("matrix multiplication shader"),
			source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
		});

		let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("matrix multiplication pipeline layout"),
			bind_group_layouts: &[&matrix_layout, &matrix_layout, &matrix_layout],
			push_constant_ranges: &[],
		});

		let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
			label: Some("matrix multiplication pipeline"),
			layout: Some(&pipeline_layout),
			module: &shader,
			entry_point: "compute",
		});

		Gpu {
			instance,
			adapter,
			device,
			queue,
			shader,
			pipeline_layout,
			pipeline,
			matrix_layout,
		}
	}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dirty {
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
	gpu: Arc<Gpu>,
	gpu_frame: Option<MatrixGPUFrame>,
}

impl Matrix {
	pub fn move_to_gpu(&mut self) {
		match &self.gpu_frame {
			Some(frame) => {
				// the GPU buffer is allocated, so we just need to make sure it's in sync with the CPU buffer.
				if frame.dirty == Dirty::Clean || frame.dirty == Dirty::CPUDirty {
					// The CPU doesn't have any changes, so we don't need to do anything.
					return;
				}
				// Copy the data to the GPU.
				let mut _encoder =
					self.gpu
						.device
						.create_command_encoder(&wgpu::CommandEncoderDescriptor {
							label: Some("matrix copy encoder"),
						});
			}
			None => {
				// The GPU buffer isn't allocated, so we need to allocate it.
				let buffer =
					self.gpu
						.device
						.create_buffer_init(&wgpu::util::BufferInitDescriptor {
							label: Some("matrix buffer"),
							contents: bytemuck::cast_slice(&self.data),
							usage: wgpu::BufferUsages::STORAGE
								| wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
						});
				let size_buffer =
					self.gpu
						.device
						.create_buffer_init(&wgpu::util::BufferInitDescriptor {
							label: Some("matrix size buffer"),
							contents: bytemuck::cast_slice(&[self.size as u32]),
							usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
						});
				let bind_group = self
					.gpu
					.device
					.create_bind_group(&wgpu::BindGroupDescriptor {
						label: Some("matrix bind group"),
						layout: &self.gpu.matrix_layout,
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

	fn init_gpu_dirty(&mut self) {
		if self.gpu_frame.is_some() {
			return;
		}

		let buffer = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
			label: Some("Matrix buffer"),
			size: (self.size * self.size * std::mem::size_of::<f32>()) as u64,
			usage: wgpu::BufferUsages::STORAGE
				| wgpu::BufferUsages::COPY_SRC
				| wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let size_buffer = self
			.gpu
			.device
			.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("matrix size buffer"),
				contents: bytemuck::cast_slice(&[self.size as u32]),
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			});
		let bind_group = self
			.gpu
			.device
			.create_bind_group(&wgpu::BindGroupDescriptor {
				label: Some("matrix bind group"),
				layout: &self.gpu.matrix_layout,
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

		let frame = MatrixGPUFrame {
			dirty: Dirty::GPUDirty,
			size_buffer,
			buffer,
			bind_group,
		};

		self.gpu_frame = Some(frame);
	}

	pub fn move_to_cpu(&mut self) {
		let buffer = match &mut self.gpu_frame {
			None => return,
			Some(frame) => {
				if frame.dirty != Dirty::CPUDirty {
					return;
				}
				frame.dirty = Dirty::Clean;
				&frame.buffer
			}
		};
		let mem_size = (self.size * self.size * std::mem::size_of::<f32>()) as u64;
		let read_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
			label: None,
			size: mem_size,
			mapped_at_creation: false,
			usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
		});
		let mut encoder = self
			.gpu
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("matrix read encoder"),
			});
		let (tx, rx) = std::sync::mpsc::channel::<()>();
		encoder.copy_buffer_to_buffer(buffer, 0, &read_buf, 0, mem_size);
		self.gpu.queue.submit(std::iter::once(encoder.finish()));
		read_buf.slice(..).map_async(wgpu::MapMode::Read, move |_| {
			tx.send(()).expect("Failed to read");
		});
		self.gpu.device.poll(wgpu::Maintain::Wait);
		rx.recv().expect("Failed to map");
		let buf_map = read_buf.slice(..).get_mapped_range();
		let float_slice = bytemuck::cast_slice::<u8, f32>(&buf_map.deref());
		self.data.clone_from_slice(float_slice);
	}

	pub fn gpu(&self) -> &Arc<Gpu> {
		&self.gpu
	}

	pub fn generate<F: FnMut(usize, usize) -> f32>(
		gpu: Arc<Gpu>,
		size: usize,
		mut factory: F,
	) -> Self {
		let mut data = Vec::with_capacity(size * size);
		for row in 0..size {
			for col in 0..size {
				data.push(factory(row, col));
			}
		}
		Self {
			size,
			data,
			gpu,
			gpu_frame: None,
		}
	}

	pub fn zeros(gpu: Arc<Gpu>, size: usize) -> Self {
		Self {
			gpu,
			size,
			gpu_frame: None,
			data: vec![0.0; size * size],
		}
	}

	pub fn eye(gpu: Arc<Gpu>, size: usize) -> Self {
		Self::generate(gpu, size, |row, col| if row == col { 1.0 } else { 0.0 })
	}
}

impl std::ops::Mul for &Matrix {
	type Output = Matrix;

	fn mul(self, rhs: Self) -> Self::Output {
		assert_eq!(self.size, rhs.size);
		assert!(
			Arc::ptr_eq(&self.gpu, &rhs.gpu),
			"matrices must be on the same GPU"
		);
		let self_frame = self
			.gpu_frame
			.as_ref()
			.expect("Matrix must be loaded on the GPU");
		let rhs_frame = rhs
			.gpu_frame
			.as_ref()
			.expect("Matrix must be loaded on the GPU");
		let mut out = Matrix::zeros(Arc::clone(&self.gpu), self.size);
		out.init_gpu_dirty();
		let out_frame = out.gpu_frame.as_ref().unwrap();
		let mut encoder = self
			.gpu
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("matrix multiplication encoder"),
			});
		let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
			label: Some("matrix multiplication pass"),
		});
		cpass.set_pipeline(&self.gpu.pipeline);
		cpass.set_bind_group(0, &out_frame.bind_group, &[]);
		cpass.set_bind_group(1, &self_frame.bind_group, &[]);
		cpass.set_bind_group(2, &rhs_frame.bind_group, &[]);
		let workgroup_num = (self.size as u32 + 15) / 16;
		cpass.dispatch_workgroups(workgroup_num, workgroup_num, 1);
		drop(cpass);
		let (tx, rx) = std::sync::mpsc::channel::<()>();
		self.gpu.queue.on_submitted_work_done(move || {
			tx.send(())
				.expect("Failed to wait for matrix multiplication");
		});
		self.gpu.queue.submit(std::iter::once(encoder.finish()));
		rx.recv().expect("Failed to wait for matrix multiplication");
		out.gpu_frame.as_mut().unwrap().dirty = Dirty::CPUDirty;
		out
	}
}

impl Debug for Matrix {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Matrix")
			.field("size", &self.size)
			.field("frame", &self.gpu_frame.as_ref().map(|f| f.dirty))
			.finish_non_exhaustive()
	}
}

impl Display for Matrix {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "[")?;
		for row in 0..self.size {
			write!(f, " [")?;
			let mut first = true;
			for col in 0..self.size {
				if first {
					write!(f, "{:.3}", self.data[row * self.size + col])?;
					first = false;
				} else {
					write!(f, " {:.3}", self.data[row * self.size + col])?;
				}
			}
			writeln!(f, "]")?;
		}
		writeln!(f, "]")?;
		Ok(())
	}
}

impl std::ops::Index<(usize, usize)> for Matrix {
	type Output = f32;

	fn index(&self, index: (usize, usize)) -> &Self::Output {
		&self.data[index.0 as usize * self.size + index.1 as usize]
	}
}
