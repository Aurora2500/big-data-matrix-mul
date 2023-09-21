#![allow(dead_code)]
mod gpu;

use std::sync::Arc;

use gpu::{Gpu, Matrix};

use rand::prelude::*;

fn main() {
	let gpu = Arc::new(pollster::block_on(Gpu::new()));

	let mut rng = rand::thread_rng();

	let size = 5792;

	let mut mat1 = Matrix::generate(Arc::clone(&gpu), size, |_, _| rng.gen());
	let mut mat2 = Matrix::generate(Arc::clone(&gpu), size, |_, _| rng.gen());

	let m_start = std::time::Instant::now();

	mat1.move_to_gpu();
	mat2.move_to_gpu();

	let m_end = std::time::Instant::now();

	let m_duration = (m_end - m_start).as_secs_f64() / 2.0;

	println!("--- setup done (took {m_duration:.3}s per matrix) ---");

	let start = std::time::Instant::now();
	let _out_mat = &mat1 * &mat2;
	let end = std::time::Instant::now();
	let duration = (end - start).as_secs_f64();

	println!("duration: {duration:.4}s",);
}
