#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1028

void matmul(float *out, float *a, float *b, size_t size)
{
	for (int row = 0; row < size; row++)
	{
		for (int col = 0; col < size; col++)
		{
			float s = 0;
			for (int k = 0; k < size; k++)
			{
				s += a[k + row * size] * b[col + k * size];
			}
			out[col + row * size] = s;
		}
	}
}

int main(void)
{
	struct timeval start, end;
	float *A = malloc(N * N * sizeof(float));
	float *B = malloc(N * N * sizeof(float));
	float *C = malloc(N * N * sizeof(float));
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i + j * N] = rand() / RAND_MAX;
			B[i + j * N] = rand() / RAND_MAX;
		}
	}

	gettimeofday(&start, NULL);

	matmul(C, A, B, N);

	gettimeofday(&end, NULL);

	printf("Time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec));
}