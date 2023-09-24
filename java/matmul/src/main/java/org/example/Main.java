package org.example;

import java.util.Random;

public class Main {
	public static void main(String[] args) {
		Random random = new Random();

		for (int i = 1; i <= 12; i++) {
			int n = (1 << i);
			Matrix a = new Matrix(n, random);
			Matrix b = new Matrix(n, random);

			long start = System.nanoTime();
			Matrix c = a.multiply(b);
			long end = System.nanoTime();
			System.out.println("Size: " + n + " | Time: " + String.format("%.6f", (end - start) * 1e-9 ));
		}
	}
}