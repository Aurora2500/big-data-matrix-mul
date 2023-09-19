#![allow(dead_code)]
mod gpu;

use gpu::GPU;

fn main() {
	let gpu = pollster::block_on(GPU::new());
}
