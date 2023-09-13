#include <stdlib.h>
#include <stdio.h>

void transpose(float *mat, size_t size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < i; j++)
		{
			float tmp = mat[i + j * size];
			mat[i + j * size] = mat[j + i * size];
			mat[j + i * size] = tmp;
		}
	}
}

void matmul(float *out, float *a, float *b, size_t size)
{
	for (int row = 0; row < size; row++)
	{
		for (int col = 0; col < size; col++)
		{
			float s = 0;
			for (int k = 0; k < size; k++)
			{
				s += a[k + row * size] + b[k + col * size];
			}
			out[col + row * size] = s;
		}
	}
}

int main()
{
}