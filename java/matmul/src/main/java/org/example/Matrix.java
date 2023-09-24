package org.example;

import java.util.Random;

public class Matrix {
	private float data[];
	private int size;

	public Matrix(int size) {
		this.size = size;
		this.data = new float[size * size];
	}

	public Matrix(int size, Random rng) {
		this(size);
		for (int i = 0; i < size * size; i++) {
			data[i] = rng.nextFloat();
		}
	}

	public float get(int row, int column) {
		return data[row * size + column];
	}

	public void set(int row, int column, float value) {
		data[row * size + column] = value;
	}

	public Matrix multiply(Matrix other) {
		if (size != other.size) {
			throw new IllegalArgumentException("Matrix dimensions do not match");
		}
		Matrix result = new Matrix(size);
		for (int row = 0; row < size; row++) {
			for (int column = 0; column < size; column++) {
				float sum = 0;
				for (int k = 0; k < size; k++) {
					sum += get(row, k) * other.get(k, column);
				}
				result.set(row, column, sum);
			}
		}
		return result;
	}
}
