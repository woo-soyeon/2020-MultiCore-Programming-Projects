#include "UpScale.h"

__global__ void Nearest_Neighborhood_Kernel(PIXEL* dLR, PIXEL *dRI, int Sampling, int row, int col, int ROW, int COL, int channels) {
	int RIx = blockDim.x * blockIdx.x + threadIdx.x;
	int RIy = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;

	int LRx = floor((float)RIx / Sampling + 0.5);
	int LRy = floor((float)RIy / Sampling + 0.5);

	if (RIy < ROW && RIx < COL) {
		dRI[RIy * (COL * channels) + RIx * (channels) + c] = dLR[LRy * (col * channels) + LRx * (channels) + c];
	}
}

__global__ void Bilinear_Kernel(PIXEL* dLR, PIXEL *dRI, int Sampling, int row, int col, int ROW, int COL, int channels) {
	int RIx = blockDim.x * blockIdx.x + threadIdx.x;
	int RIy = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;

	int LRx = RIx / Sampling;
	int LRy = RIy / Sampling;

	if (RIy < ROW && RIx < COL) {
		int m = RIy % Sampling;
		int n = RIx % Sampling;
		if (m == 0 && n == 0) {
			dRI[RIy * (COL * channels) + RIx * (channels) + c] = dLR[LRy * (col * channels) + LRx * (channels) + c];
		}
		else {
			if (RIy == ROW - 1) {
				LRy--;
				m = Sampling - m;
			}
			if (RIx == COL - 1) {
				LRx--;
				n = Sampling - n;
			}

			PIXEL v1 = dLR[(LRy + 0) * (col * channels) + (LRx + 0) * (channels) + c];
			PIXEL v2 = dLR[(LRy + 0) * (col * channels) + (LRx + 1) * (channels) + c];
			PIXEL v3 = dLR[(LRy + 1) * (col * channels) + (LRx + 0) * (channels) + c];
			PIXEL v4 = dLR[(LRy + 1) * (col * channels) + (LRx + 1) * (channels) + c];

			double w1 = (double)((Sampling - m) * (Sampling - n)) / (Sampling * Sampling);
			double w2 = (double)((Sampling - m) * (n)) / (Sampling * Sampling);
			double w3 = (double)((m) * (Sampling - n)) / (Sampling * Sampling);
			double w4 = (double)((m) * (n)) / (Sampling * Sampling);
			dRI[RIy * (COL * channels) + RIx * (channels) + c] = ((v1 * w1) + (v2 * w2) + (v3 * w3) + (v4 * w4));
		}
	}
}

__device__ __host__ double clamp(const double value, const double min, const double max) {
	return value < min ? min : max < value ? max : value;
}

__device__ __host__ double cubic(PIXEL v1, PIXEL v2, PIXEL v3, PIXEL v4, double d) {
	double result = ((-v1 + (3 * v2) - (3 * v3) + v4) * pow(d, 3)) +
		(((2 * v1) - (5 * v2) + (4 * v3) - v4) * pow(d, 2)) +
		(v3 - v1) * d +
		(2 * v2);
	result /= 2;

	return clamp(result, 0, 255);
}

__global__ void Bicubic_Kernel_Y(PIXEL* dLR, PIXEL *dRI, int Sampling, int row, int col, int ROW, int COL, int channels) {
	int RIx = blockDim.x * blockIdx.x + threadIdx.x;
	int RIy = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;

	int LRx = RIx / Sampling;
	int LRy = RIy / Sampling;

	if (RIy < ROW && RIx < COL) {
		int m = RIy % Sampling;
		int n = RIx % Sampling;
		if (n != 0) return;

		if (m == 0 && n == 0) {
			dRI[RIy * (COL * channels) + RIx * (channels) + c] = dLR[LRy * (col * channels) + LRx * (channels) + c];
		}
		else {
			double d = (double)m / Sampling;
			if (LRy == 0) {
				LRy += 1; d -= 1;
			}
			else if (LRy == row - 2) {
				LRy -= 1; d += 1;
			}
			PIXEL v1 = dLR[(LRy - 1) * (col * channels) + LRx * (channels) + c];
			PIXEL v2 = dLR[(LRy + 0) * (col * channels) + LRx * (channels) + c];
			PIXEL v3 = dLR[(LRy + 1) * (col * channels) + LRx * (channels) + c];
			PIXEL v4 = dLR[(LRy + 2) * (col * channels) + LRx * (channels) + c];

			dRI[RIy * (COL * channels) + RIx * (channels)+c] = cubic(v1, v2, v3, v4, d);
		}
	}
}

__global__ void Bicubic_Kernel_X(PIXEL* dLR, PIXEL *dRI, int Sampling, int row, int col, int ROW, int COL, int channels) {
	int RIx = blockDim.x * blockIdx.x + threadIdx.x;
	int RIy = blockDim.y * blockIdx.y + threadIdx.y;
	int c = blockDim.z * blockIdx.z + threadIdx.z;

	int LRx = RIx / Sampling;
	int LRy = RIy / Sampling;

	if (RIy < ROW && RIx < COL) {
		int n = RIx % Sampling;
		int tRIx = RIx;
		if (n == 0) return;

		double d = (double)n / Sampling;
		if (tRIx <= Sampling) {
			tRIx += Sampling; d -= 1;
		}
		else if (tRIx >= COL - Sampling) {
			tRIx -= Sampling; d += 1;
		}
		PIXEL v1 = dRI[RIy * (COL * channels) + ((tRIx - n) - (1 * Sampling)) * (channels) + c];
		PIXEL v2 = dRI[RIy * (COL * channels) + ((tRIx - n) + (0 * Sampling)) * (channels) + c];
		PIXEL v3 = dRI[RIy * (COL * channels) + ((tRIx - n) + (1 * Sampling)) * (channels) + c];
		PIXEL v4 = dRI[RIy * (COL * channels) + ((tRIx - n) + (2 * Sampling)) * (channels) + c];

		dRI[RIy * (COL * channels) + RIx * (channels)+c] = cubic(v1, v2, v3, v4, d);
	}
}

Mat GPU_Call(Mat LR, int Sampling, int mod, dim3 gridDim, dim3 blockDim,DS_timer* timer, int timerID) {
	int row = LR.rows;
	int col = LR.cols;
	int channels = LR.channels();

	int ROW = row * Sampling - (Sampling - 1);
	int COL = col * Sampling - (Sampling - 1);

	Mat RI = Mat::zeros(ROW, COL, LR.type());

	int LR_Size = row * col * channels * sizeof(PIXEL);
	int RI_Size = ROW * COL * channels * sizeof(PIXEL);

	PIXEL* dLR = NULL;
	PIXEL* dRI = NULL;
	cudaMalloc(&dLR, LR_Size);
	cudaMemset(dLR, 0, LR_Size);
	cudaMalloc(&dRI, RI_Size);
	cudaMemset(dRI, 0, RI_Size);

	cudaMemcpy(dLR, LR.data, LR_Size, cudaMemcpyHostToDevice);

	timer->onTimer(timerID);
	switch (mod) {
	case Iterpolation::NN:
		Nearest_Neighborhood_Kernel <<<gridDim, blockDim>>>(dLR, dRI, Sampling, row, col, ROW, COL, channels);
		break;
	case Iterpolation::BILINEAR:
		Bilinear_Kernel <<<gridDim, blockDim>>>(dLR, dRI, Sampling, row, col, ROW, COL, channels);
		break;
	case Iterpolation::BICUBIC:
		Bicubic_Kernel_Y <<<gridDim, blockDim>>>(dLR, dRI, Sampling, row, col, ROW, COL, channels);
		cudaDeviceSynchronize();
		Bicubic_Kernel_X <<<gridDim, blockDim >>>(dLR, dRI, Sampling, row, col, ROW, COL, channels);
		break;
	case Iterpolation::DEFAULT:
		Nearest_Neighborhood_Kernel << <gridDim, blockDim >> >(dLR, dRI, Sampling, row, col, ROW, COL, channels);
		break;
	}
	cudaDeviceSynchronize();
	timer->offTimer(timerID);

	cudaMemcpy(RI.data, dRI, RI_Size, cudaMemcpyDeviceToHost);

	/*
	printf("GPU\n");
	for (int c = 0; c < channels; c++) {
		for (int y = 0; y < RI.rows; y++) {
			for (int x = 0; x < RI.cols; x++) {
				printf("%3d ", (int)RI.at<Vec3b>(y, x)[c]);
			}
			printf("\n");
		}
		printf("\n");
	}*/

	cudaFree(dLR);
	cudaFree(dRI);
	return RI;
}

Mat Nearest_Neighborhood(Mat LR, int Sampling) { //최단입점 보간법
	int row = LR.rows;
	int col = LR.cols;
	int channels = LR.channels();

	int ROW = row * Sampling - (Sampling - 1);
	int COL = col * Sampling - (Sampling - 1);
	Mat RI = Mat::zeros(ROW, COL, LR.type());

	
	for (int RIy = 0; RIy < ROW; RIy++) {
		for (int RIx = 0; RIx < COL; RIx++) {
			int LRy = floor((float)RIy / Sampling + 0.5);
			int LRx = floor((float)RIx / Sampling + 0.5);
			RI.at<Vec3b>(RIy, RIx) = LR.at<Vec3b>(LRy, LRx);
		}
	}
	return RI;
}

Mat Bilinear(Mat LR, int Sampling) {  //선형보간법
	int row = LR.rows;
	int col = LR.cols;
	int channels = LR.channels();

	int ROW = row * Sampling - (Sampling - 1);
	int COL = col * Sampling - (Sampling - 1);

	Mat RI = Mat::zeros(ROW, COL, LR.type());

	
	for (int RIy = 0; RIy < ROW; RIy++) {
		for (int RIx = 0; RIx < COL; RIx++) {
			for (int c = 0; c < channels; c++) {
				int LRy = (float)RIy / Sampling;
				int LRx = (float)RIx / Sampling;
				int m = RIy % Sampling;
				int n = RIx % Sampling;
				if (m == 0 && n == 0) {
					RI.at<Vec3b>(RIy, RIx) = LR.at<Vec3b>(LRy, LRx);
					break;
				}
				else {
					if (RIy == ROW - 1) {
						LRy--;
						m = Sampling - m;
					}
					if (RIx == COL - 1) {
						LRx--;
						n = Sampling - n;
					}

					PIXEL v1 = LR.at<Vec3b>((LRy + 0), (LRx + 0))[c];
					PIXEL v2 = LR.at<Vec3b>((LRy + 0), (LRx + 1))[c];
					PIXEL v3 = LR.at<Vec3b>((LRy + 1), (LRx + 0))[c];
					PIXEL v4 = LR.at<Vec3b>((LRy + 1), (LRx + 1))[c];

					double w1 = (double)((Sampling - m) * (Sampling - n)) / (Sampling * Sampling);
					double w2 = (double)((Sampling - m) * (n)) / (Sampling * Sampling);
					double w3 = (double)((m) * (Sampling - n)) / (Sampling * Sampling);
					double w4 = (double)((m) * (n)) / (Sampling * Sampling);
					RI.at<Vec3b>(RIy, RIx)[c] = ((v1 * w1) + (v2 * w2) + (v3 * w3) + (v4 * w4));
				}
			}
		}
	}
	return RI;
}

Mat Bicubic(Mat LR, int Sampling) {
	int row = LR.rows;
	int col = LR.cols;
	int channels = LR.channels();

	int ROW = row * Sampling - (Sampling - 1);
	int COL = col * Sampling - (Sampling - 1);

	Mat RI = Mat::zeros(ROW, COL, LR.type());
	
	for (int RIy = 0; RIy < ROW; RIy++) {
		for (int RIx = 0; RIx < COL; RIx += Sampling) {
			for (int c = 0; c < channels; c++) {
				int LRy = (float)RIy / Sampling;
				int LRx = (float)RIx / Sampling;
				int m = RIy % Sampling;
				int n = RIx % Sampling;
				if (m == 0 && n == 0) {
					RI.at<Vec3b>(RIy, RIx) = LR.at<Vec3b>(LRy, LRx);
					break;
				}

				double d = (double)m / Sampling;
				if (LRy == 0) {
					LRy += 1; d -= 1;
				}
				else if (LRy == row - 2) {
					LRy -= 1; d += 1;
				}
				PIXEL v1 = LR.at<Vec3b>(LRy - 1, LRx)[c];
				PIXEL v2 = LR.at<Vec3b>(LRy + 0, LRx)[c];
				PIXEL v3 = LR.at<Vec3b>(LRy + 1, LRx)[c];
				PIXEL v4 = LR.at<Vec3b>(LRy + 2, LRx)[c];
				
				RI.at<Vec3b>(RIy, RIx)[c] = cubic(v1, v2, v3, v4, d);
			}
		}
	}

	for (int RIy = 0; RIy < ROW; RIy++) {
		for (int RIx = 1; RIx < COL; RIx++) {
			for (int c = 0; c < channels; c++) {
				int n = RIx % Sampling;
				int tRIx = RIx;
				if (n == 0) break;

				double d = (double)n / Sampling;
				if (tRIx <= Sampling) {
					tRIx += Sampling; d -= 1;
				}
				else if (tRIx >= COL - Sampling) {
					tRIx -= Sampling; d += 1;
				}
				PIXEL v1 = RI.at<Vec3b>(RIy, (tRIx - n) - (1 * Sampling))[c];
				PIXEL v2 = RI.at<Vec3b>(RIy, (tRIx - n) + (0 * Sampling))[c];
				PIXEL v3 = RI.at<Vec3b>(RIy, (tRIx - n) + (1 * Sampling))[c];
				PIXEL v4 = RI.at<Vec3b>(RIy, (tRIx - n) + (2 * Sampling))[c];

				RI.at<Vec3b>(RIy, RIx)[c] = cubic(v1, v2, v3, v4, d);
			}
		}
	}
	return RI;
}

Mat CPU_Call(Mat LR, int Sampling, int mod) {
	Mat result;
	switch (mod) {
	case Iterpolation::NN:
		result = Nearest_Neighborhood(LR, Sampling);
		break;
	case Iterpolation::BILINEAR:
		result = Bilinear(LR, Sampling);
		break;
	case Iterpolation::BICUBIC:
		result = Bicubic(LR, Sampling);
		break;
	case Iterpolation::DEFAULT:
		result = Nearest_Neighborhood(LR, Sampling);
		break;
	}
	return result;
}