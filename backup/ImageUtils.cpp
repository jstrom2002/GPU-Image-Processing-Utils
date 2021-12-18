#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "ImageUtils.h"
#include "StringUtils.h"

#include "util/image_processing_util.h"

#include "CurvSolver.h"//for quadricCurvature

#include "surfaceBlur.h"

//for now, OCR is done via the Tesseract library
#include "Tesseract-OCR.h"

//for Bernard's 'copyFill1D()'
#include <arrayfire.h>
#include <boost/stacktrace.hpp>


#include <opencv2/opencv.hpp>

af::array binary_threshold(const af::array& in, float thresholdValue) {
	//from arrayfire's example code: https://github.com/arrayfire/arrayfire/blob/master/examples/image_processing/adaptive_thresholding.cpp

	int channels = in.dims(2);
	af::array ret_val = in.copy();
	if (channels > 1) ret_val = colorSpace(in, AF_GRAY, AF_RGB);
	ret_val =
		(ret_val < thresholdValue) * 0.0f + 255.0f * (ret_val > thresholdValue);
	return ret_val;
}

std::string textRecognition(af::array im, std::string expected_string) {
	///*		Text Recognition using Tesseract -- might not work if VS version < 2019.
	//*		This code example is from https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
	//*		NOTE: Tesseract requires that training data be stored locally, in a folder found using a system environment variable named TESSDATA_PREFIX.
	//*		This data is available online at: https://github.com/tesseract-ocr/tessdata		*/

	//// Create Tesseract object -- faster if only initialized once
	//tesseract::TessBaseAPI* ocr = new tesseract::TessBaseAPI();

	//// Set parameters to match our specs
	////ocr->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");//whitelist chars we need to look for here
	//ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);//use LSTM neural network
	//ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);//rail data is only a single line of alphanumeric characters

	////initialize all variables, load and set image in Tesseract engine
	std::string outText = "";
	//ocr->SetImage(
	//	im.host<unsigned char>(),
	//	im.dims(1),
	//	im.dims(0),
	//	3,
	//	sizeof(float)
	//);

	//// Run Tesseract OCR on image
	//outText = std::string(ocr->GetUTF8Text());

	//// Clean up memory leaks
	//ocr->Clear();
	//ocr->End();

	return outText;
}


void loadRanger3ToAF(std::string filename, af::array& dst) {
	//load data from raw .bin file
	Ranger3::RawData rw;
	std::vector<uint8_t> buffer;
	buffer.reserve(Ranger3::RANGEFILESIZE);
	if (Ranger3::FileToBuffer(filename, buffer.data(), Ranger3::RANGEFILESIZE) != 0) { return; }

	rw.Range.clear();
	rw.Range.reserve(Ranger3::SCANSIZE * Ranger3::SCANRESOLUTION);
	rw.Range.insert(rw.Range.begin(),
		static_cast<const Ranger3::pixel_t*>(static_cast<const void*>(buffer.data())),
		static_cast<const Ranger3::pixel_t*>(static_cast<const void*>(buffer.data())) + Ranger3::BLOCKELEMENTS
	);

	//delete[] buffer;
	buffer.clear();

	//convert buffer to af::array
	if (rw.Range.size() <= 0) { std::cout << "Error! Ranger3 data is not initialized yet" << std::endl;	return; }
	dst = af::array(1344, 4096, rw.Range.data());

	//free allocated memory
	rw.Range.clear();
	rw.Mark.clear();
	if (!rw.Range.empty()) { std::cout << "deallocation failed!\n"; }
	if (!rw.Mark.empty()) { std::cout << "deallocation failed!\n"; }
}

void GaussianBlur(af::array src, af::array& dst, int kernelX, int kernelY, double sig_r, double sig_c) {
	//from arrayfire's example code: https://github.com/arrayfire/arrayfire/blob/master/examples/image_processing/image_demo.cpp

	af::array gauss_k;
	af::gaussianKernel(kernelX, kernelY, sig_r, sig_c);
	dst = af::convolve(src, gauss_k);
}

void sharpen(af::array src, af::array& dst, int kernelX, int kernelY, double sigmaX, double sigmaY) {
	af::array blurredImage;
	GaussianBlur(src, blurredImage, kernelX, kernelY, sigmaX, sigmaY);
	dst = src - blurredImage;
}

void normalize(af::array& img) {
	img /= af::max(img).scalar<double>();
	threshold0to1(img, img);
}

void makePositiveValued(af::array& mat) {
	double min_ = af::min(mat).scalar<double>();
	if (min_ < 0) {
		mat += min_;
	}
}

af::array rangefilt(af::array pI) {
	/* Adapted from Seth's TAS1ProcessRail code, from the rangefilt.cpp source file. This function mirrors the rangefilt::get() method */

	//af::array pI input matrix, neighborhood size must be odd
	af::array pout = af::array(pI.dims(0), pI.dims(1), pI.type());		//output matrix	
	af::array pH = af::constant(1, 3, 3, u8);						//filter matrix, neighborhood must be binary -- default is 3x3 matrix but it can be user-supplied

	af::array reflectH;

	reflectH = af::flip(pH, 0);//flip matrix vertically
	//reflectH = arma::reshape(reflectH, arma::size(pH));
	af::array rhcv = af::array(pH.dims(0), pH.dims(1), u8);
	af::array hcv = af::array(pH.dims(0), pH.dims(1), u8);
	for (int i = 0; i < pH.dims(0) - 1; i++) {
		for (int j = 0; j < pH.dims(1) - 1; j++) {
			rhcv(i, j) = reflectH(i, j).scalar<uint8_t>();
			hcv(i, j) = pH(i, j).scalar<uint8_t>();
		}
	}
	af::array dilateI, erodeI;
	image_processing_util::morphClose1D(erodeI, pI, 2, 1);
	bool logical = 1;

	// check to see if the pI matrix is a binary matrix
	for (int i = 0; i < pI.dims(0) - 1; i++) {
		for (int j = 0; j < pI.dims(1) - 1; j++) {
			if (pI(i, j).scalar<float>() != 0 && pI(i, j).scalar<float>() != 1) {
				logical = 0;
				break;
			}
		}
	}

	// if so, save elements where dilation is greater than erosion
	if (logical) {
		af::resize(pout, dilateI.dims(0), dilateI.dims(1));
		for (int i = 0; i < pout.dims(0) - 1; i++) {
			for (int j = 0; j < pout.dims(1) - 1; j++) {
				pout(i, j) = dilateI(i, j).scalar<float>() > erodeI(i, j).scalar<float>();
			}
		}
		return pout;
	}

	// else, elements are set to the value of the difference between the dilation and erosion values
	else {
		af::array out = dilateI - erodeI;

		af::resize(pout, out.dims(0), out.dims(1));
		for (int i = 0; i < out.dims(0) - 1; i++) {
			for (int j = 0; j < out.dims(1) - 1; j++) {
				pout(i, j) = out(i, j).scalar<float>();
			}
		}
		return pout;
	}
}

void imwrite(std::string filename, af::array img) {
	img *= (256.0 * 256.0) - 1;
	af::saveImage(filename.c_str(), img.as(u16));
}

void imshow(af::array img, int w, int h) {
	af::Window window(w, h, "");
	do {
		window.image(img);
	} while (!window.close());
}

af::array imread(std::string filename) {
	return af::loadImageNative(filename.c_str());
}

void threshold0to1(af::array src, af::array& dst) {
	int channels = src.dims(2);
	dst = src.copy();
	if (channels > 1) dst = af::colorSpace(src, AF_GRAY, AF_RGB);
	dst = (dst < 0) * 0.0f + 1.0f * (dst > 1);
}

void thresholdNeg1to1(af::array src, af::array& dst) {
	int channels = src.dims(2);
	dst = src.copy();
	if (channels > 1) dst = af::colorSpace(src, AF_GRAY, AF_RGB);
	dst = (dst < -1) * -1.0f + 1.0f * (dst > 1);
}

void getChannel(af::array& img, af::array& dst, unsigned int channel) {
	dst = img(af::span, af::span, channel);
}


void rangeDataToPointCloud(std::string filename, af::array& dst) {
	//converted from Bernard's rw::shape_analysis code


	//load data from raw .bin file
	Ranger3::RawData data;
	std::vector<uint8_t> buffer;
	buffer.reserve(Ranger3::RANGEFILESIZE);
	if (Ranger3::FileToBuffer(filename, buffer.data(), Ranger3::RANGEFILESIZE) != 0) { return; }

	data.Range.clear();
	data.Range.reserve(Ranger3::SCANSIZE * Ranger3::SCANRESOLUTION);
	data.Range.insert(data.Range.begin(),
		static_cast<const Ranger3::pixel_t*>(static_cast<const void*>(buffer.data())),
		static_cast<const Ranger3::pixel_t*>(static_cast<const void*>(buffer.data())) + Ranger3::BLOCKELEMENTS
	);


	if (data.Range.empty()) { return; }
	af::array buffer2 = af::array(Ranger3::BLOCKELEMENTS, reinterpret_cast<const unsigned short*>(data.Range.data()));

	////OLD CODE -- Use homogeneous coodinates (4D)
	//dst = af::join(0,
	//	af::moddims(af::flat(af::transpose(af::tile(af::seq(Ranger3::SCANSIZE), 1, Ranger3::SCANRESOLUTION))), 1, 1, Ranger3::BLOCKELEMENTS),
	//	af::tile(af::moddims(af::seq(Ranger3::SCANRESOLUTION), 1, 1, Ranger3::SCANRESOLUTION), 1, 1, Ranger3::SCANSIZE),
	//	af::moddims(buffer2.as(f32), 1, 1, Ranger3::BLOCKELEMENTS),
	//	af::constant(1.0f, 1, 1, Ranger3::BLOCKELEMENTS)
	//);

	//NEW CODE -- 3D coordinates
	/*dst = af::join(0,
		af::moddims(af::flat(af::transpose(af::tile(af::seq(Ranger3::SCANSIZE), 1, Ranger3::SCANRESOLUTION))), 1, 1, Ranger3::BLOCKELEMENTS),
		af::tile(af::moddims(af::seq(Ranger3::SCANRESOLUTION), 1, 1, Ranger3::SCANRESOLUTION), 1, 1, Ranger3::SCANSIZE),
		af::moddims(buffer2.as(f32), 1, 1, Ranger3::BLOCKELEMENTS)
	);*/

	//AS 2D image
	dst = af::moddims(buffer2.as(u16), 1344, 4096).as(f32);
}

void quadricCurvature(af::array& src2, af::array& normals, af::array& curvature, af::array& curvatureNaNs, int radius, int max_its, bool inpaintFirst) {
	int frame_width = 4096;
	int frame_height = 1344;

	af::array src;
	//loadRanger3ToAF("Y:\\TT_Rail_Services\\TAS_RAIL_3007\\Data\\20181120\\RailWEB\\20181120093116\\W20181120204315\\2\\2R000001.dat", src);
	rangeDataToPointCloud("Y:\\TT_Rail_Services\\TAS_RAIL_3007\\Data\\20181120\\RailWEB\\20181120093116\\W20181120204315\\2\\2R000001.dat", src);//Bernard's code

	printMatrixInfo(src);


	//initialize solver class for quadric curvature
	CURVATURE::CurvSolver* solver = new CURVATURE::CurvSolver(src.dims(1), src.dims(0), radius, max_its);

	////inpaint image
	//if (inpaintFirst) {
	//	af::array srcAF = af::array(1344, 4096, f32);
	//	srcAF = src.copy();
	//	af::array mask = srcAF > 0;
	//	af::array outAF;
	//	image_processing_util::copyFill1D(src, srcAF, mask, 1);
	//}

	////transfer image data to 'depth' array
	uint16_t* depth = new uint16_t[1344 * 4096];
	depth = src.as(u16).T().host<uint16_t>();
	solver->copy_rgbd((uint16_t*)depth);

	std::unique_ptr<float> h_normals(new float[src.dims(1) * src.dims(0) * 3]);
	std::unique_ptr<float> h_curvature(new float[src.dims(1) * src.dims(0) * 3]);
	std::unique_ptr<float> h_coords(new float[src.dims(1) * src.dims(0) * 4]);

	solver->compute();

	solver->get_curvature(h_curvature.get());
	solver->get_normals(h_normals.get());

	int n = 1344 * 4096 * 3;
	std::vector<float> vec1;
	std::vector<float> vec2;
	vec1.resize(n);
	vec2.resize(n);
	for (int i = 0; i < n; ++i) {
		vec1[i] = h_normals.get()[i];
		vec2[i] = h_curvature.get()[i];
	}

	////TEST -- check results with OpenCV
	////=================================
	cv::Mat normals2 = cv::Mat(vec1).reshape(3, src.dims(0));
	cv::Mat curvature2 = cv::Mat(vec2).reshape(3, src.dims(0));
	cv::Mat gray, displayMat;
	std::vector<cv::Mat> bgr;
	cv::split(curvature2, bgr);
	gray = (cv::abs(bgr[0]) + cv::abs(bgr[1])) * 0.5;		//get mean curvature

	//normalize 'gray'
	double matrixMin, matrixMax;
	cv::minMaxLoc(gray, &matrixMin, &matrixMax);
	gray += cv::Scalar::all(matrixMin);
	cv::minMaxLoc(gray, &matrixMin, &matrixMax);
	gray *= 1.0 / matrixMax;
	cv::threshold(gray, gray, 0, 1, cv::THRESH_TOZERO);		//values < 0 are thresholded to 0
	cv::threshold(gray, gray, 1, 1, cv::THRESH_TRUNC);		//values > 1 are thresholded to 1

	curvature2 = gray;

	//cv::resize(normals2, displayMat, cv::Size(normals2.cols * 0.5, normals2.rows * 0.5));
	//cv::imshow("TEST", displayMat);
	//cv::waitKey();
	//cv::resize(curvature2, displayMat, cv::Size(curvature2.cols * 0.5, curvature2.rows * 0.5));
	//cv::imshow("TEST", displayMat);
	//cv::waitKey();
	////=========================================================================================




	//cv::cvtColor(normals2, normals2, cv::COLOR_BGR2RGB);
	//cv::cvtColor(curvature2, curvature2, cv::COLOR_BGR2RGB);

	//normals2 = normals2.reshape(0);
	//curvature2 = curvature2.reshape(0);

	//normals = af::array(normals2.cols, normals2.rows, 3, normals2.ptr<float>(0));//creates pixel packing issues
	//curvature = af::array(curvature2.cols, curvature2.rows, curvature2.ptr<float>(0));

	//normals = af::array(normals2.cols, normals2.rows, 3, normals2.data).as(f32).T();//less of an issue with pixel packing
	//curvature = af::array(curvature2.cols, curvature2.rows, curvature2.data).as(f32).T();

	//normals = af::array(normals2.cols, normals2.rows, 3, normals2.data).T();
	//curvature = af::array(curvature2.cols, curvature2.rows, curvature2.data).T();

	//normals = af::array(src.dims(0), src.dims(1), 3, vec.data());
	//curvature = af::array(src.dims(0), src.dims(1), 3, vec2.data());



	normals2.convertTo(normals2, CV_32F);
	curvature2.convertTo(curvature2, CV_32F);
	normals = af::array(normals2.cols * normals2.rows * 3, normals2.data).as(f32);//less of an issue with pixel packing
	curvature = af::array(curvature2.cols * curvature2.rows, curvature2.data).as(f32);
	normals = af::moddims(normals, normals2.cols, normals2.rows, 3).T();
	curvature = af::moddims(curvature, curvature2.cols, curvature2.rows).T();






	printMatrixInfo(normals);
	printMatrixInfo(curvature);

	imshow(normals);
	imshow(curvature);
	//imwrite("normals1.png", normals);
	//imwrite("curvature1.png", curvature);
	af::saveImage("normals1.png", normals);
	af::saveImage("curvature1.png", curvature);

	getChannel(curvature, curvatureNaNs, 2);//get NaNs produced by the calculation from channel 2 of the curvature image

	//get NaNs from calculation
	curvatureNaNs *= -1;//note: principal curvature sign is flipped by this implementation
	curvatureNaNs = binary_threshold(curvatureNaNs, 0.0001);

	//calculate mean curvature
	af::array B, G;
	getChannel(curvature, B, 0);
	getChannel(curvature, G, 1);
	curvature = (B + G) * -0.5;//note: principal curvature sign is flipped by this implementation

	printMatrixInfo(normals);
	printMatrixInfo(curvature);
	printMatrixInfo(curvatureNaNs);



	//free allocated memory	
	delete solver;
	//depth.clear();
	//h_curvature.clear();
	//h_normals.clear();
}

void surfaceBlur(af::array src, af::array& dat, int radius, int threshold, unsigned int method) {
	SurfaceBlur sb;
	sb.surfaceBlur(src, dat, radius, threshold, method);
}

std::string typeToString(af::dtype t) {
	switch (t) {
	case f32:
		return "f32";
	case u8:
		return "u8";
	case u16:
		return "u16";
	case b8:
		return "b8";
	case s32:
		return "s32";
	case c32:
		return "c32";
	case u32:
		return "u32";
	case f64:
		return "f64";
	case c64:
		return "c64";
	case u64:
		return "u64";
	case s16:
		return "s16";
	}
	return "";
}

void printMatrixInfo(af::array src) {
	std::cout << "type: " << typeToString(src.type()) << std::endl;
	std::cout << "bytes: " << src.bytes() << std::endl;

	std::cout << "dims: ";
	for (int i = 0; i < src.dims().ndims(); ++i) {
		std::cout << src.dims(i) << ",";
	}
	std::cout << std::endl;

	double min_, max_;
	unsigned locMax, locMin;
	af::min(&min_, &locMin, src);
	af::max(&max_, &locMax, src);

	std::cout << "range: [" << min_ << "," << max_ << "]\n" << std::endl;
}