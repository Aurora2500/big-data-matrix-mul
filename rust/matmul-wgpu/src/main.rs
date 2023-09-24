#![allow(dead_code)]
mod gpu;

use std::sync::Arc;

use gpu::{Gpu, Matrix};

use rand::prelude::*;

fn main() {
	let gpu = Arc::new(pollster::block_on(Gpu::new()));

	let mut rng = rand::thread_rng();

	let max_size = 5792;

	let mat1 = Matrix::generate(Arc::clone(&gpu), max_size, |_, _| rng.gen());
	let mat2 = Matrix::generate(Arc::clone(&gpu), max_size, |_, _| rng.gen());

	println!("Size,Time");
	let sizes = 12;
	for size in (1..=sizes).map(|n| (2u32).pow(n) as usize) {
		let mut left = Matrix::generate(Arc::clone(&gpu), size, |col, row| mat1[(col, row)]);
		let mut right = Matrix::generate(Arc::clone(&gpu), size, |col, row| mat2[(col, row)]);

		left.move_to_gpu();
		right.move_to_gpu();

		let start = std::time::Instant::now();
		let _ = &left * &right;
		let end = std::time::Instant::now();

		println!("{},{}", size, (end - start).as_secs_f64());
		std::thread::sleep(std::time::Duration::from_millis(700));
	}
}
