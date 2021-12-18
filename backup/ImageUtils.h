/**
 * \file	ImageUtils.h.
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \brief	Declares the image utilities class.
 */
#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#pragma once
#include "stdafx.h"

#include <arrayfire.h>
#include "../../Ranger3 Utils/Ranger3 Utils/Ranger3Data.h"

 /*
  * \fn	        binary_threshold(const af::array& in, float thresholdValue);
  *
  * \brief	    binary threshold of an array where values < threshold == 0, values >= threshold == 255
  *
  * \author	    Jeff Strom
  * \date   	2020-1-31
  *
  * \param 	    in                  array to be thresholded
  * \param 	    thresholdValue      threshold value
  */
af::array binary_threshold(const af::array& in, float thresholdValue);

/*
 * \fn	    std::string TextRecognition(std::string filename, std::string expected_string = "");
 *
 * \brief	Tesseract text recognition
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param 	filename	   	            Filename of the file.
 * \param 	expected_string	(Optional)  The expected string.
 *
 * \returns	A std::string.
 */
std::string textRecognition(af::array im, std::string expected_string = "");

/*
 * \fn      loadRanger3ToAF(std::string filename, af::array& dst);
 *
 * \brief	load binary Ranger3 .dat file and output it as a af::array
 *
 * \author	Jeff Strom
 * \date	2020-1-30
 *
 * \param 	filename file to open and read
 * \param 	dst af::array that will contain the final image loaded from the Ranger3 binary file
 *
 */
void loadRanger3ToAF(std::string filename, af::array& dst);

/*
 * \fn	invert(af::array src, af::array& dst);
 *
 * \brief	inverts the color values of a matrix from high to low based upon image bit depth (ie for CV_16, 65535->0, 0->65535)
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param 	src source image matrix
 * \param	dst desination image matrix
 *
 */
void GaussianBlur(af::array src, af::array& dst, int kernelX, int kernelY, double sig_r = 0, double sig_c = 0);
/*
 * \fn		sharpen(af::array src, af::array& dst, cv::Size kernelSize, double sigmaX, double sigmaY = 0)
 *
 * \brief	sharpens and image by subtracting a blurred version of it from the original
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param	src original source image matrix
 * \param	dst desination image matrix
 * \param	kernelSize size of kernel for GaussianBlur filter
 * \param	sigmaX modifier for GaussianBlur filter in the X direction
 * \param	sigmaY modifier for GaussianBlur filter in the Y direction
 */
void sharpen(af::array src, af::array& dst, int kernelX, int kernelY, double sigmaX, double sigmaY);

/*
 * \fn		normalize(af::array& mat);
 *
 * \brief	normalizes elements of float matrix to range [0,1]
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param	dst desination image matrix
 */
void normalize(af::array& mat);

/*
 * \fn		makePositiveDefinite(af::array& mat);
 *
 * \brief	converts elements of matrix to positive values by shifting all values up by the lowest non-zero value (ie the minimum value of the af::array)
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param	mat in/out image matrix
 */
void makePositiveValued(af::array& mat);

/*
 * \fn		threshold0to1(af::array src, af::array& dst);
 *
 * \brief	threshold a matrix to values in OpenCV's float range, which is [0,1], to eliminate errors when these values are converted or used.
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param	src input image matrix
 * \param	dst output image matrix
 */
void threshold0to1(af::array, af::array& dst);

/*
 * \fn		thresholdNeg1to1(af::array src, af::array& dst);
 *
 * \brief	threshold an image to values in [-1,1]
 *
 * \author	Jeff Strom
 * \date	2019-12-13
 *
 * \param	src input image
 * \param	dst output image
 */
void thresholdNeg1to1(af::array src, af::array& dst);

/*
 * \fn		    rangefilt(af::array pI);
 *
 * \brief	    filters by range using erosion and dilation (from Seth's TAS1ProcessRail code)
 *
 * \author	    Jeff Strom
 * \date	    2019-12-13
 *
 * \param	    pI input matrix to be filtered
 */
af::array rangefilt(af::array pI);


/** \fn			imwrite(std::string filename, af::array img);
 *
 *	\brief		saves a af::array with proper formatting.
 *
 *  \author		Jeff Strom
 *	\date		2019-12-13
 */
void imwrite(std::string filename, af::array img);

/** \fn			imshow(std::string filename, af::array img);
 *
 *	\brief		display an image in a window, similar to MATLAB and OpenCV's 'imshow()' function
 *
 *  \author		Jeff Strom
 *	\date		2019-12-13
 */
void imshow(af::array img, int w = 1024, int h = 720);

af::array imread(std::string filename);


/** \fn			getChannel(af::array& img, af::array& dst, unsigned int channel);
 *
 *	\brief		helper function that uses cv::split to split color image by channel, then returns the desired channel
 *
 *  \author		Jeff Strom
 *	\date		2019-12-13
 */
void getChannel(af::array& img, af::array& dst, unsigned int channel);

/** \fn			quadricCurvature(af::array& src, af::array& normals, af::array& curvature, af::array& curvatureNaNs, int radius, int max_its, bool inpaintFirst);
 *
 *	\brief		performs GPU calculation of quadric curvature and returns af::array's for normals, mean curvature, and NaN values calculated during curvature calc.
 *
 *  \author		Jeff Strom
 *	\date		2020-1-31
 */
void quadricCurvature(af::array& src, af::array& normals, af::array& curvature, af::array& curvatureNaNs, int radius, int max_its, bool inpaintFirst);

/** \fn			surfaceBlur(af::array src, af::array& dst, int radius, int threshold, unsigned int method = 4);
 *
 *	\brief		helper function to apply the surfaceBlur class found in surfaceBlur.h
 *
 *  \author		Jeff Strom
 *	\date		2020-1-31
 */
void surfaceBlur(af::array src, af::array& dst, int radius, int threshold, unsigned int method = 4);


std::string typeToString(af::dtype t);

void printMatrixInfo(af::array src);
#endif