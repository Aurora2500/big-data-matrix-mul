#![allow(dead_code)]
mod gpu;

use std::{sync::Arc, time::Duration};

use gpu::{Gpu, Matrix};

use rand::prelude::*;

fn main() {
	let gpu = Arc::new(pollster::block_on(Gpu::new()));

	let mut rng = rand::thread_rng();

	let mut mat1 = Matrix::generate(Arc::clone(&gpu), 4096, |_, _| rng.gen());
	let mut mat2 = Matrix::generate(Arc::clone(&gpu), 4096, |_, _| rng.gen());

	mat1.move_to_gpu();
	mat2.move_to_gpu();
	let mut v = Vec::new();

	println!("--- setup done ---");

	for i in 0..30 {
		std::thread::sleep(Duration::from_millis(900));
		let start = std::time::Instant::now();
		let _out_mat = &mat1 * &mat2;
		let end = std::time::Instant::now();
		let duration = (end - start).as_secs_f64();
		println!("{i:>3}: {duration}");
		v.push((end - start).as_secs_f32());
	}

	let mut sum = 0.0;
	for i in v.iter() {
		sum += i;
	}

	sum /= v.len() as f32;

	println!("{}s", sum);
}
