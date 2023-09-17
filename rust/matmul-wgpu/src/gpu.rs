pub struct GPU {
	instance: wgpu::Instance,
	adapter: wgpu::Adapter,
	device: wgpu::Device,
	queue: wgpu::Queue,
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
		};
	}
}
