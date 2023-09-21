 @group(0)
 @binding(0)
 var<uniform> size: u32;

 @group(0)
 @binding(1)
 var<storage, read_write> mat_out: array<f32>;

 @group(1)
 @binding(1)
 var<storage, read_write> mat_left: array<f32>;

 @group(2)
 @binding(1)
 var<storage, read_write> mat_right: array<f32>;

@compute @workgroup_size(16, 16)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>)
{
	let x = gid.x;
	let y = gid.y;
	if (x >= size || y >= size)
	{
		return;
	}
	var sum = 0.0;
	for (var k = 0u; k < size; k++)
	{
		let a = mat_left[y * size + k];
		let b = mat_right[k * size + x];
		sum += a * b;
	}
	mat_out[y * size + x] = sum;
}