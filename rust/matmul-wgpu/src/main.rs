use wgpu::util::DeviceExt;

async fn compute() {
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

	let matrix_bind_group_layout =
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("Matrix bind group layout"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::COMPUTE,
				ty: wgpu::BindingType::Buffer { ty: (), has_dynamic_offset: (), min_binding_size: () }
				count: None,
			}],
		});
}

fn main() {
	pollster::block_on(compute());
}
