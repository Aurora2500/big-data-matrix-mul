
struct Matrix
{
	size: u32,
	data: array<f32>,
}

@group(0)
@binding(0)
var<storage, read_write> mat_output: Matrix;

@group(1)
@binding(0)
var<storage, read> mat_left: Matrix;

@group(2)
@binding(0)
var<storage, read> mat_right: Matrix;


@compute @workgroup_size(32, 32)
fn compute(@builtin(global_invocation_id) gid: vec3<u32>)
{
	let x = gid.x;
	let y = gid.y;
	if (x >= mat_output.size || y >= mat_output.size)
	{
		return;
	}
	let size = mat_output.size;
	var sum = 0.0;
	for (var k = 0u; k < size; k++)
	{
		let a = mat_left.data[y * size + k];
		let b = mat_right.data[k * size + x];
		sum += a * b;
	}
	mat_output.data[y * size + x] = sum;
}