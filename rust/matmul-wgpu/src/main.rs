#![allow(dead_code)]
mod gpu;
mod matrix;

use gpu::GPU;

fn main() {
	let gpu = pollster::block_on(GPU::new());
}
