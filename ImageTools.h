#pragma once
#include <arrayfire.h>

void harrisCorner(af::array& src, af::array& dst) {
	//get 1st derivatives
	af::array ix, iy;
	af::grad(ix, iy, src);

	//get 2nd order derivatives
	af::array ixx = ix * ix;
	af::array ixy = ix * iy;
	af::array iyy = iy * iy;

	// Compute a Gaussian kernel with standard deviation of 1.0 and length of 5 pixels
	// These values can be changed to use a smaller or larger window
	int window_x = 5;
	int window_y = 5;
	double sigma_x = 1.0;
	double sigma_y = 1.0;
	af::array gauss_filt = af::gaussianKernel(window_x, window_y, sigma_x, sigma_y);

	// Filter second-order derivatives with Gaussian kernel computed previously
	ixx = af::convolve(ixx, gauss_filt);
	ixy = af::convolve(ixy, gauss_filt);
	iyy = af::convolve(iyy, gauss_filt);

	// Calculate trace
	af::array tr = ixx + iyy;
	// Calculate determinant
	af::array det = ixx * iyy - ixy * ixy;

	// Calculate Harris response
	af::array response = det - 0.04f * (tr * tr);

	// Gets maximum response for each 3x3 neighborhood
	int filt_x = 3;
	int filt_y = 3;
	af::array max_resp = af::maxfilt(response, filt_x, filt_y);

	// Discard responses that are not greater than threshold
	double threshold = 1e5;
	af::array corners = response > threshold;
	corners = corners * response;

	// Discard responses that are not equal to maximum neighborhood response,
	// scale them to original response value
	dst = corners = (corners == max_resp) * corners;
}

af::array binary_threshold(const af::array& in, float thresholdValue) {
	//from arrayfire's example code: https://github.com/arrayfire/arrayfire/blob/master/examples/image_processing/adaptive_thresholding.cpp

	int channels = in.dims(2);
	af::array ret_val = in.copy();
	if (channels > 1) ret_val = colorSpace(in, AF_GRAY, AF_RGB);
	ret_val =
		(ret_val < thresholdValue) * 0.0f + 255.0f * (ret_val > thresholdValue);
	return ret_val;
}


//The following functions are from ArrayFire's own code for morphological operations,
//found at: http://arrayfire.org/docs/image_processing_2morphing_8cpp-example.htm
af::array morphopen(const af::array& img, const af::array& mask)
{
	return af::dilate(af::erode(img, mask), mask);
}
af::array morphclose(const af::array& img, const af::array& mask)
{
	return af::erode(af::dilate(img, mask), mask);
}
af::array morphgrad(const af::array& img, const af::array& mask)
{
	return (af::dilate(img, mask) - af::erode(img, mask));
}
af::array tophat(const af::array& img, const af::array& mask)
{
	return (img - morphopen(img, mask));
}
af::array bottomhat(const af::array& img, const af::array& mask)
{
	return (morphclose(img, mask) - img);
}
af::array border(const af::array& img, const int left, const int right,
	const int top, const int bottom,
	const float value = 0.0)
{
	if ((int)img.dims(0) < (top + bottom))
		printf("input does not have enough rows\n");
	if ((int)img.dims(1) < (left + right))
		fprintf(stderr, "input does not have enough columns\n");
	af::dim4 imgDims = img.dims();
	af::array ret = af::constant(value, imgDims);
	ret(af::seq(top, imgDims[0] - bottom), af::seq(left, imgDims[1] - right), af::span, af::span) =
		img(af::seq(top, imgDims[0] - bottom), af::seq(left, imgDims[1] - right), af::span, af::span);
	return ret;
}
af::array border(const af::array& img, const int w, const int h,
	const float value = 0.0)
{
	return border(img, w, w, h, h, value);
}
af::array border(const af::array& img, const int size, const float value = 0.0)
{
	return border(img, size, size, size, size, value);
}
