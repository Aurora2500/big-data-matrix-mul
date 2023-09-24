#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "stopwatch.h"

#include <xmmintrin.h>

#define N_THREADS 16

struct transpose_job
{
	size_t size, start_idx, len;
	float *mat;
};

void *transpose_task(void *job_ptr)
{
	struct transpose_job *job = (struct transpose_job *)job_ptr;
	size_t size = job->size;
	size_t start_idx = job->start_idx;
	size_t len = job->len;
	float *mat = job->mat;

	/*
		The task is done only for the upper triangular part of the matrix.
		As the main diagnoal is not changed, and each transpose of the upper triangular part deals
		with the corresponding lower triangular part.

		In a 5 by 5 matrix, this is what the indices are for each position:

		x 0 1 3 6
		x x 2 4 7
		x x x 5 8
		x x x x 9
		x x x x x

		Thus a matrix of size N has T_(n-1) indices, where T_n is the nth triangular number.
		So a size N matrix has indices going from 0 to ((N-1)^2 + (N-1)) / 2 - 1.
	*/

	// source: https://math.stackexchange.com/questions/1417579/largest-triangular-number-less-than-a-given-natural-number
	size_t row = (size_t)((1 + sqrt(1 + 8 * start_idx)) / 2);
	size_t col_start = row * (row - 1) / 2;
	size_t col = start_idx - col_start;

	for (size_t i = 0; i < len; i++)
	{
		float tmp = mat[row + col * size];
		mat[row + col * size] = mat[col + row * size];
		mat[col + row * size] = tmp;

		col++;
		if (col == row)
		{
			row++;
			col = 0;
		}
	}
	return NULL;
}

void transpose(float *mat, size_t size)
{
	size_t T_n1 = ((size - 1) * size) / 2;
	struct transpose_job tasks[N_THREADS];
	pthread_t threads[N_THREADS - 1];
	size_t tasks_per_thread = T_n1 / N_THREADS;
	size_t leftover_tasks = T_n1 % N_THREADS;
	for (size_t i = 0; i < N_THREADS; i++)
	{
		tasks[i].size = size;
		tasks[i].mat = mat;
		tasks[i].start_idx = i * tasks_per_thread + (i < leftover_tasks ? i : leftover_tasks);
		tasks[i].len = T_n1 / N_THREADS + (i < leftover_tasks ? 1 : 0);
		if (i < N_THREADS - 1)
		{
			pthread_create(&threads[i], NULL, transpose_task, &tasks[i]);
		}
		else
		{
			transpose_task(&tasks[i]);
		}
	}
	for (int i = 0; i < N_THREADS - 1; i++)
	{
		pthread_join(threads[i], NULL);
	}
}

struct matmul_job
{
	size_t size, start_idx, len;
	float *left, *right, *result;
};

void *matmul_task(void *ptr)
{
	struct matmul_job *job = (struct matmul_job *)ptr;
	size_t size = job->size;
	size_t start_idx = job->start_idx;
	size_t len = job->len;
	float *left = job->left;
	float *right = job->right;
	float *result = job->result;

	for (size_t idx = start_idx; idx < (start_idx + len); idx++)
	{
		size_t row = idx / size;
		size_t col = idx % size;
		float s = 0;
		size_t k = 0;
		if (size >= 4)
			for (; k < size - 3; k += 4)
			{
				float simd_res[4];
				__m128 l, r, prod;
				l = _mm_load_ps(&left[k + row * size]);
				r = _mm_load_ps(&right[k + col * size]);
				prod = _mm_mul_ps(l, r);
				_mm_store_ps(simd_res, prod);
				s += simd_res[0] + simd_res[1] + simd_res[2] + simd_res[3];
			}
		for (; k < size; k++)
		{
			s += left[k + row * size] * right[k + col * size];
		}
		result[col + row * size] = s;
	}
	return NULL;
}

void matmul(float *out, float *a, float *b, size_t size)
{
	struct matmul_job tasks[N_THREADS];
	pthread_t threads[N_THREADS - 1];
	size_t tasks_per_thread = size * size / N_THREADS;
	size_t leftover_tasks = size * size % N_THREADS;

	for (size_t i = 0; i < N_THREADS; i++)
	{
		tasks[i].size = size;
		tasks[i].left = a;
		tasks[i].right = b;
		tasks[i].result = out;
		tasks[i].start_idx = i * tasks_per_thread + (i < leftover_tasks ? i : leftover_tasks);
		tasks[i].len = size * size / N_THREADS + (i < leftover_tasks ? 1 : 0);
		if (i < N_THREADS - 1)
		{
			pthread_create(&threads[i], NULL, matmul_task, &tasks[i]);
		}
		else
		{
			matmul_task(&tasks[i]);
		}
	}
	for (int i = 0; i < N_THREADS - 1; i++)
	{
		pthread_join(threads[i], NULL);
	}
}

int main()
{
	const int runs = 10;
	const int sizes = 12;
	// double times[runs][sizes];

	printf("Size,Time\n");
	fflush(stdout);

	int n = 1;
	for (int i = 1; i <= sizes; i++)
	{
		n *= 2;
		double duration = 0.0;
		for (size_t k = 0; k < runs; k++)
		{
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

			// Prevent compiler from optimizing away the computation
			asm volatile("" : : "g"(C) : "memory");

			duration += stopwatch_elapsed(&sw) / runs;
		}
		printf("%d,%f\n", n, duration);
		fflush(stdout);
	}
}