#pragma once
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "DS_timer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;

#define PIXEL unsigned char

enum Iterpolation {
	NN = 0x00,
	BILINEAR = 0x01,
	BICUBIC = 0x10,
	DEFAULT = 0x11
};

Mat GPU_Call(Mat LR, int Sampling, int mod);

