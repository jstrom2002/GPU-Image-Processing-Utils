#pragma comment(lib, "af.lib")

#include <iostream>
#include <random>
#include <ctime>
#include <cmath>
#include "ImageTools.h"
#include "HoughTF.h"

void isolateJointBar(af::array& src, af::array& dst) {
	//heavy blur in the y-direction in image space
	af::array kern = af::gaussianKernel(1, 9, 1.0, 9.0);
	af::array blurred = af::convolve(src, kern);

	//get gradient in y-direction
	af::array dx, dy;
	af::grad(dx, dy, blurred);
	
	//binarize image ahead of morphological op
	dy = binary_threshold(dy, 0.1);

	//do morphological opening to eliminate disconnected lines caused by noise
	int maskSize = 3;
	af::array mask = af::constant(1,maskSize,maskSize,dy.type());//af::array(3,3, arr).as(f32);
	dst = morphopen(dy, mask).as(f32);
}


int main() {
	//af::array img = af::loadImageNative("jointBar.png").as(f32);
	af::array img = af::loadImageNative("jointBarEDITED.png").as(f32);//removed ties/noise on bottom, necessary for good detection
	//af::array img = af::loadImageNative("noJointBar.png").as(f32);
	
	af::Window window(1074, 720, "");
	do {window.image(img.as(f32));} 
	while (!window.close());

	// Convert the image from RGB to gray-scale if necessary
	if (img.dims(2) > 1) {
		af::colorSpace(img, AF_GRAY, AF_RGB);
	}

	//isolate joint bar
	isolateJointBar(img, img);

	//find Hough lines
	af::array binary = binary_threshold(img, 0.1);
	std::vector<LineParameter> detectedLine;
	HTLineDetection(binary.as(u8).host<unsigned char>(), detectedLine, binary.dims(0), binary.dims(1));

	   
	do { window.image(img.as(f32)); } while (!window.close());

	af::saveImage("detected.png", img);

	std::cout << "number of lines: " << detectedLine.size() << std::endl;
	std::vector<float> x_vals, y_vals;
	srand(clock());
	for (int i = 0; i < detectedLine.size(); ++i) {
		std::cout << "line dist: " << detectedLine[i].distance << ",  line angle: " << detectedLine[i].angle << std::endl;
		
		//find points using parametric equation of the Hough line
		for (int i = 0; i < 10; ++i) {
			int y_val_ = img.dims(0) + 1;
			int x_val_ = 0;
			while (y_val_ >= img.dims(0) || y_val_ < 0) {
				x_val_ = rand() % (int)sqrt((double)pow(img.dims(1),2) + (double)pow(img.dims(0),2));
				y_val_ = detectedLine[i].PolarToCartesian(x_val_);
			}
			x_vals.push_back(x_val_);
			y_vals.push_back(y_val_);
			std::cout << "point on line: (" << x_val_ << "," << y_val_ << ")\n";
		}
	}
	af::array lines_x = af::array(x_vals.size(), x_vals.data()).as(f32);
	af::array lines_y = af::array(y_vals.size(), y_vals.data()).as(f32);
	af::join(1, lines_x, lines_y);

	do { window.scatter(lines_x, AF_MARKER_CROSS); } while (!window.close());
}