#include <stdio.h>
#include <stdlib.h>
#include "stopwatch.h"

void matmul(float *out, float *a, float *b, size_t size)
{
	for (size_t row = 0; row < size; row++)
	{
		fprintf(stderr, "%zu / %zu\n", row, size);
		fflush(stderr);
		for (size_t col = 0; col < size; col++)
		{
			float s = 0;
			for (size_t k = 0; k < size; k++)
			{
				s += a[k + row * size] * b[col + k * size];
			}
			out[col + row * size] = s;
		}
	}
}

int main(void)
{
	//const int runs = 10;
	const int sizes = 12;
	//double times[runs][sizes];

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
		matmul(C, A, B, n);
		stopwatch_stop(&sw);

		asm volatile("": : "g"(C) : "memory");

		double duration = stopwatch_elapsed(&sw);

		printf("%d,%f\n", n, duration);
		fflush(stdout);
	}
}