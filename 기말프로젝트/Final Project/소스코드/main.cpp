#include "UpScale.h"

#define BLOCK_SIZE (16)

bool CompareMat(Mat A, Mat B);

int main(int argc, char** argv) {
	if (argc < 3) {
		printf("It requires three arguments\n");
		printf("Usage: Extuction_file number a, b, n\n");
		return -1;
	}
	DS_timer timer(6);
	timer.setTimerName(0, "NN_CPU      ");
	timer.setTimerName(1, "Bilinear_CPU");
	timer.setTimerName(2, "Bicubic_CPU ");
	timer.setTimerName(3, "NN_GPU      ");
	timer.setTimerName(4, "Bilinear_GPU");
	timer.setTimerName(5, "Bicubic_GPU ");

	timer.initTimers();

	Mat image = imread(argv[1], IMREAD_COLOR);
	int Sampling = atoi(argv[2]);
	
	if (image.empty()) {
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	//* 1. CPU code *//
	timer.onTimer(0);
	Mat nn_CPU       = CPU_Call(image, Sampling, Iterpolation::NN);
	timer.offTimer(0);
	timer.onTimer(1);
	Mat bilinear_CPU = CPU_Call(image, Sampling, Iterpolation::BILINEAR);
	timer.offTimer(1);
	timer.onTimer(2);
	Mat bicubic_CPU  = CPU_Call(image, Sampling, Iterpolation::BICUBIC);
	timer.offTimer(2);

	//* 2. CUDA code *//
	dim3 gridDim(ceil((float)(image.cols * Sampling - (Sampling - 1)) / BLOCK_SIZE), ceil((float)(image.rows * Sampling - (Sampling - 1)) / BLOCK_SIZE), 1);
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 3);

	Mat nn_GPU       = GPU_Call(image, Sampling, Iterpolation::NN, gridDim, blockDim, &timer, 3);
	Mat bilinear_GPU = GPU_Call(image, Sampling, Iterpolation::BILINEAR, gridDim, blockDim, &timer, 4);
	Mat bicubic_GPU  = GPU_Call(image, Sampling, Iterpolation::BICUBIC, gridDim, blockDim, &timer, 5);


	//* 3. Result Checking code *//
	if (CompareMat(nn_CPU, nn_GPU)) {
		printf("The Nearest_Neighborhood is good matched!\n");
		imwrite("Nearest_Neighborhood.jpg", nn_GPU);
		//imshow("Nearest_Neighborhood", nn_GPU);
	}
	else printf("The Nearest_Neighborhood is not matched!\n");
	
	if (CompareMat(bilinear_CPU, bilinear_GPU)) {
		printf("The Bilinear is good matched!\n");
		imwrite("Bilinear.jpg", bilinear_GPU);
		//imshow("Bilinear", bilinear_GPU);
	}
	else printf("The Bilinear is not matched!\n");
	
	if (CompareMat(bicubic_CPU, bicubic_GPU)) {
		printf("The Bicubic is good matched!\n");
		imwrite("Bicubic.jpg", bicubic_CPU);
		//imshow("Bicubic", bicubic_CPU);
	}
	else printf("The Bicubic is not matched!\n");

	timer.printTimer();

	waitKey(0);
	system("pause");
	return 0;
}

bool CompareMat(Mat A, Mat B) {
	bool result = true;

	int row = A.rows;
	int col = A.cols;
	int channel = A.channels();

	if (row == B.rows && col == B.cols && channel == B.channels()) {
		for (int y = 0; y < row; y++) {
			for (int x = 0; x < col; x++) {
				for (int c = 0; c < channel; c++) {
					if (abs(A.at<Vec3b>(y, x)[c] - B.at<Vec3b>(y, x)[c]) > 1) {
						result = false;
						break;
					}
				}
			}
		}
	}
	else {
		return false;
	}
	return result;
}