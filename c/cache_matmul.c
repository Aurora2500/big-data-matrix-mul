#include <stdlib.h>
#include <stdio.h>
#include "stopwatch.h"

void transpose(float *mat, size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		for (size_t j = 0; j < i; j++)
		{
			float tmp = mat[i + j * size];
			mat[i + j * size] = mat[j + i * size];
			mat[j + i * size] = tmp;
		}
	}
}

void matmul(float *out, float *a, float *b, size_t size)
{
	for (size_t row = 0; row < size; row++)
	{
		for (size_t col = 0; col < size; col++)
		{
			float s = 0;
			for (size_t k = 0; k < size; k++)
			{
				s += a[k + row * size] + b[k + col * size];
			}
			out[col + row * size] = s;
		}
	}
}

int main()
{
	const int sizes = 12;

	printf("Size,Time\n");
	fflush(stdout);
	int n = 1;
	for(int i = 1; i <= sizes; i++) {
		n *= 2;
		struct stopwatch sw;
		float *A = malloc(n * n * sizeof(float));
		float *B = malloc(n * n * sizeof(float));
		float *C = malloc(n * n * sizeof(float));
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				A[i + j * n] = rand() / RAND_MAX;
				B[i + j * n] = rand() / RAND_MAX;
			}
		}

		stopwatch_start(&sw);
		transpose(B, n);
		matmul(C, A, B, n);
		stopwatch_stop(&sw);

		asm volatile("": : "g"(C) : "memory");

		double duration = stopwatch_elapsed(&sw);

		printf("%d,%f\n", n, duration);
		fflush(stdout);
	}
}