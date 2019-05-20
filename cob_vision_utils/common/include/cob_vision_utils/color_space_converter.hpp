/*!
 *****************************************************************
 * @file color_space_converter.hpp
 *
 * @brief This file is the header that provides the class that can convert from the RGB color space into several
 * other user-defined color spaces.
 *
 * @details This file is the header for the ::ColorSpaceConverter class. This class provides the functionality to
 * convert an image, which is in the RGB color space, to several user-defined color spaces. The input RGB-image is
 * assumed to be in CV_U8C3 format.
 *
 *****************************************************************
 *
 * @note Copyright (c) 2017 Fraunhofer Institute for Manufacturing Engineering and Automation (IPA)
 * @note Project name: none
 * @note ROS stack name: cob_perception_common
 * @note ROS package name: cob_vision_utils
 *
 * @author Author: Florian Jordan
 *
 * @date Date of creation: 11.2017
 *
 *
 *****************************************************************
 *
 * Copyright 2017 Fraunhofer Institute for Manufacturing Engineering
 * and Automation (IPA)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************/

#pragma once

// opencv includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std includes
#include <iostream>

/**
 * @brief The ColorSpaceConvertionType enum is the enum that encodes, which convertion should be applied on an RGB image. It
 * can be used for the api-function in ::ColorSpaceConverter for a convenient transformation.
 * @todo move xyY to latter position --> s.t. it can be ignored, is bad
 */
enum ColorSpaceConvertionType
{
	UNDEFINED=-1, RGB2RGB=0, RGB2rgb=1, RGB2RGBII=2, RGB2XYZ=3, RGB2xyY=4, RGB2XYZII=5, RGB2HSV=6, RGB2Lab=7, RGB2Opponent=8, RGB2I3=9,
	RGB2C3=10, RGB2LSLM=11, RGB2Luv=12, RGB2l3=13, RGB2ZRG=14
};

/**
 * @brief The ColorSpaceConverter class is a helper class that can be used to transform from one color space into another.
 */
class ColorSpaceConverter
{
protected:

	/**
	 * @brief vertical_divisons_ stores how many vertical (along x-axis) division should be done, when seperating the image
	 * into several patches (used for several transformations).
	 */
	uint vertical_divisons_;

	/**
	 * @brief vertical_divisons_ stores how many horizontal (along y-axis) division should be done, when seperating the image
	 * into several patches (used for several transformations).
	 */
	uint horizontal_divisons_;

	/**
	 * @brief calculatePatchStatistics is the function that calculates the mean and variance for each channel over the given patch.
	 * @details This function can be used to calculate the mean and variance for each color channel over the given patch. The results
	 * are stored inside a vector, which will be of size 3, that stores pairs of doubles. The first double shows the mean and the second
	 * double shows the variance for each color channel. The function also takes the class of vectors that is stored inside the patch
	 * as an template argument (e.g. cv::Vec3b for CV_8U).
	 * @param patch: The patch for which the mean and variance should be computed for every channel.
	 * @param channel_statistics: Vector that carries both statistical values for each channel (first: mean, second: variance).
	 */
	template<typename ValueClass>
	void calculatePatchStatistics(const cv::Mat& patch, std::vector<std::pair<double, double> >& channel_statistics);

public:

	/**
	 * @brief ColorSpaceConverter is the constructor that directly sets the parameters that control in how many patches an image
	 * should be divided in some transformations.
	 * @param vertical_divisons: The number of vertical divisons that should be set.
	 * @param horizontal_divisons: The number of horizontal divisons that should be set.
	 * @param standardize_output_image: The Boolean, showing if the resulting output should be standardized or not.
	 */
	ColorSpaceConverter(const uint& vertical_divisons=0, const uint& horizontal_divisons=0, const bool standardize_output_image=false);

	/**
	 * @brief transformColorSpace is an api-function that can be used to transform an RGB image to another color space.
	 * @details This function provides a simple api to the several functions of this class, to provide a convenient way
	 * of transforming an RGB image into another color space. The input image needs to be an RGB image in the CV_8UC3 format,
	 * the resulting output image is in the CV32_FC3 format in the desired color space. The desired color space can be
	 * determined by a parameter, using the ::ColorSpaceConvertionType enum.
	 * @param input: The to-be-converted RGB image, in CV_8UC3 format.
	 * @param output: The resulting transformed image in CV_32FC3 format.
	 * @param type: The ::ColorSpaceConvertionType, showing to which color space the RGB image shall be converted.
	 * @return 0 if the convertion succeeded, -1 if not.
	 */
	int transformColorSpace(const cv::Mat& input, cv::Mat& output, const ColorSpaceConvertionType& type);

	/**
	 * @brief RGB2ZRG converts an image from the RGB-space into the ZRG space.
	 * @details The ZRG color space is a hybrid color space, that consists of the Z-component of the XYZ space and the RG-components of
	 * the RGB-space, see @cite yang_color_2010.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the ZRG space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2ZRG(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2rgb converts an image from the RGB-space into the rgb space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the rgb space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2rgb(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2RGBII converts an image from the RGB-space into the R_IIB_II_G_II space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @details See @cite yang_color_2010 for details
	 * @param output: The resulting output-image in the rgb space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2RGBII(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2XYZ converts an image from the RGB-space into the XYZ space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the XYZ space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2XYZ(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2xyY converts an image from the RGB-space into the xyY (XYZ space with X/Y-channel being normalized
	 * and including Y itself) space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the xyY space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2xyY(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2XYZII converts an image from the RGB-space into the X_IIY_IIZ_II space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the X_IIY_IIZ_II space (resulting format: CV_32FC3).
	 * @details See @cite yang_color_2010 for details
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2XYZII(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2HSV converts an image from the RGB-space into the HSV space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the HSV space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2HSV(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2Lab converts an image from the RGB-space into the CIEL*a*b* space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the CIEL*a*b* space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2Lab(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2Opponent converts an image from the RGB-space into the opponent-color-space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the opponent-color-space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2Opponent(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGBI3 converts an image from the RGB-space into the I_1I_2I_3 space.

	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the I_1I_2I_3 space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2I3(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGBC3 converts an image from the RGB-space into the C_1C_2C_3 space.
	 * @details See @cite gevers_color-based_1999 for details on this color space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the C_1C_2C_3 space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2C3(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGBLuv converts an image from the RGB-space into the CIE-LUV space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the CIE-LUV space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2Luv(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2LSLM converts an image from the RGB-space into the LSLM space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the LSLM space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2LSLM(const cv::Mat& input, cv::Mat& output);

	/**
	 * @brief RGB2LSLM converts an image from the RGB-space into the LSLM space.
	 * @details See @cite gevers_color-based_1999 for details on this color space.
	 * @param input: The input image in RGB-format (required format: CV_8UC3).
	 * @param output: The resulting output-image in the LSLM space (resulting format: CV_32FC3).
	 * @note Throws a CV_Error, if the input requirements are not met.
	 */
	void RGB2l3(const cv::Mat& input, cv::Mat& output);
};
