/*!
 *****************************************************************
 * @file histogram_computer.hpp
 *
 * @brief This file is the header that provides a class that is capable of computing a 1D histogram out of
 * a given color image. The image is assumed to be in CV_32FC3 format.
 *
 * @details
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
 * @date Date of creation: 12.2017
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

// std includes
#include <mutex>
#include <vector>
#include <tuple>

// opencv includes
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * @brief channelmerged_channel_one_region_number_ represents the number of regions in which the first channel
 * should be divided when using channelmerged color histograms.
 */
const static size_t channelmerged_channel_one_region_number_=7;

/**
 * @brief channelmerged_channel_two_region_number_ represents the number of regions in which the second channel
 * should be divided when using channelmerged color histograms.
 */
const static size_t channelmerged_channel_two_region_number_=7;

/**
 * @brief channelmerged_channel_three_region_number_ represents the number of regions in which the third channel
 * should be divided when using channelmerged color histograms.
 */
const static size_t channelmerged_channel_three_region_number_=7;

/**
 * @brief channelwise_channel_one_region_number_ represents the number of regions in which the first channel
 * should be divided when using channelwise color histograms.
 */
const static size_t channelwise_channel_one_region_number_=20;

/**
 * @brief channelwise_channel_two_region_number_ represents the number of regions in which the second channel
 * should be divided when using channelwise color histograms.
 */
const static size_t channelwise_channel_two_region_number_=20;

/**
 * @brief channelwise_channel_three_region_number_ represents the number of regions in which the third channel
 * should be divided when using channelwise color histograms.
 */
const static size_t channelwise_channel_three_region_number_=20;

/**
 * @brief number_of_together_regions_ is the total number of regions, set by ::channelmerged_channel_one_region_number_,
 * ::channelmerged_channel_two_region_number_ and ::channelmerged_channel_three_region_number_, if you use
 * channelmerged histograms.
 */
const static size_t number_of_together_regions_ = channelmerged_channel_one_region_number_*channelmerged_channel_two_region_number_*channelmerged_channel_three_region_number_;

/**
 * @brief number_of_separate_regions_ is the total number of regions, set by ::channelwise_channel_one_region_number_,
 * ::channelwise_channel_two_region_number_ and ::channelwise_channel_three_region_number_, if you use the channel-wise histogram.
 */
const static size_t number_of_separate_regions_ = channelwise_channel_one_region_number_+channelwise_channel_two_region_number_+channelwise_channel_three_region_number_;

/**
 * @brief operator <= is the definition of the comparsion operator for two OpenCV 3D vectors. It simply checks if
 * every element of this vector is smaller or equal than the corresponding element of the given vector.
 * @param rhs: The vector that defines the upper bounds of each element.
 * @return true, if every element of this vector is smaller or equal than the corresponding element in the given vector.
 */
template <typename _Tp>
bool operator<=(const cv::Vec<_Tp, 3>& lhs, const cv::Vec<_Tp, 3>& rhs)
{
	// catch float precision when checking the equality
	return ((lhs[0]<rhs[0] || std::abs(lhs[0]-rhs[0])<=1e-6) && (lhs[1]<rhs[1] || std::abs(lhs[1]-rhs[1])<=1e-6)
			&& (lhs[2]<rhs[2] || std::abs(lhs[2]-rhs[2])<=1e-6));
}

/**
 * @brief operator >= is the definition of the comparsion operator for two OpenCV 3D vectors. It simply checks if
 * every element of this vector is bigger or equal than the corresponding element of the given vector.
 * @param rhs: The vector that defines the lower bounds of each element.
 * @return true, if every element of this vector is bigger or equal than the corresponding element in the given vector.
 */
template <typename _Tp>
bool operator>=(const cv::Vec<_Tp, 3>& lhs, const cv::Vec<_Tp, 3>& rhs)
{
	// catch float precision when checking the equality
	return ((lhs[0]>rhs[0] || std::abs(lhs[0]-rhs[0])<=1e-6) && (lhs[1]>rhs[1] || std::abs(lhs[1]-rhs[1])<=1e-6)
			&& (lhs[2]>rhs[2] || std::abs(lhs[2]-rhs[2])<=1e-6));
}

/**
 * @brief The ColorHistogramComputer class is capable of comouting a 1D histogram out of a given color image. The given image is
 * assumed to be in the CV_32FC3 format.
 * @details This class provides the functionality to compute a 1D histogram out of a given color image in CV_32FC3 format. To do so
 * it divides the 3 color channels into a specified number of regions. It then goes through the image pixels and checks, in which
 * ranges the channels of this specific pixel are and adds an entry in this bin. After every pixel has been assigned to one bin,
 * the algorithm is also capable of normalizing the color histogram.
 */
class ColorHistogramComputer
{
protected:

	/**
	 * @brief color_space_channel_ranges_ stores the min/max values in the different channels for the different defined color spaces.
	 * @details For each color space a pair of two 3-dimensional column vectors are stored. The first vector represents the minimal values
	 * each channel can take and the second vector represents the maximal values in each channel.
	 */
	std::vector<std::pair<cv::Vec3f, cv::Vec3f> > color_space_channel_ranges_;

	/**
	 * @brief access_histogram_bin_mutex_ is the mutex that is locked, whenever one bin of the histogram is accessed, when it is computed
	 * in parallel.
	 */
	mutable std::mutex access_histogram_bin_mutex_;

	/**
	 * @brief number_of_threads_ is the maximal number of threads the histogram computer is allowed to open.
	 */
        size_t number_of_threads_;

	/**
	 * @brief normalizeHistogram is the function that normalizes the given histogram to unit length, using the L1 norm.
	 * @param histogram: The histogram that should be normalized.
	 */
	void normalizeHistogram(std::vector<float>& histogram) const;

public:

	/**
	 * @brief ColorHistogramComputer is the constructor of the class, that automatically sets the initial parameters of the class.
	 */
        ColorHistogramComputer(const size_t& number_of_threads=4);

	/**
	 * @brief setNumberOfThreads is the setter function for the number of threads this computer is allowed to open.
	 * @param number_of_threads: The allowed number of threads for this computer.
	 */
	void setNumberOfThreads(const size_t& number_of_threads);


	/**
	 * @brief getRegionsForColorSpace is the function that is called whenever the histogram bin regions for a
	 * certain color space shall be computed.
	 * @param regions: Vector that carries the computed bin regions of the color space.
	 * @param color_space: Indicator for the color space for which the regions shall be computed.
	 * @param c1_regions: The number of regions in which the first channel of the colorspace shall be divided.
	 * @param c2_regions: The number of regions in which the second channel of the colorspace shall be divided.
	 * @param c3_regions: The number of regions in which the third channel of the colorspace shall be divided.
	 */
	void getRegionsForColorSpace(std::vector<std::pair<cv::Vec3f, cv::Vec3f> >& regions,
								 const size_t& color_space, const size_t& c1_regions,
								 const size_t& c2_regions, const size_t& c3_regions) const;

	/**
	 * @brief calculateHistogram is the function that takes an image in a arbitrary color space and computes a 1D histogram from it by
	 * dividing the channels into several regions.
	 * @details This function divides the defined color space into several subspaces (depending on the set number of channel regions) and
	 * assigns each pixel in a given image to one of these subspaces. This results into a 1D histogram, that represents the colors in the
	 * image. The corresponding color regions will be constructed by iterating over the channel ranges of the color space, with the first
	 * channel in the outer loop and the third channel in the inner loop.
	 * The following color spaces are recognized:
	 *
	 *	- 0: RGB
	 *	- 1: rgb (channel normalized RGB)
	 *	- 2: R_IIG_IIB_II (across-component normalized RGB, see @cite yang_color_2010)
	 *	- 3: XYZ
	 *	- 4: xyY
	 *	- 5: X_IIY_IIZ_II (across-component normalized XYZ, see @cite yang_color_2010)
	 *	- 6: HSV
	 *	- 7: CIEL*a*b*
	 *	- 8: opponent color space
	 *	- 9: I_1I_2I_3
	 *	- 10: C_1C_2C_3
	 *	- 11: LSLM, see @cite yang_color_2010
	 *	- 12: CIE-LUV
	 *	- 13: l_1l_2l_3, (see @cite gevers_color-based_1999 for details)
	 *
	 * @param image: The image, for which the histogram should be computed.
	 * @param color_space: Parameter that indicates in which color space the given image is encoded (different color spaces have different
	 * ranges in the channels).
	 * @param histogram: The resulting 1D histogram.
	 * @param region_of_interest: Optional pointer parameter to a binary image (CV_8UC1) that shows with white pixels which pixels should
	 * be taken into account for the histogram computation.
	 * @param regions: The regions to which the pixels have been assigned during the procedure, depends on the defined color space. Optional
	 * parameter that can be used to get the regions to which the pixels have been assigned.
	 * @param region_numbers: Optional pointer to a cv::Vec3i object that stores how many regions shall be used
	 * per channel.
	 */
	void calculateHistogram(const cv::Mat& image, const size_t& color_space, std::vector<float>& histogram,
							const cv::Mat* region_of_interest=0, std::vector<std::pair<cv::Vec3f, cv::Vec3f> >* bin_regions=0,
							const cv::Vec3i* region_numbers=0) const;

	/**
	 * @brief calculateChannelWiseHistogram is the function that takes an image in a arbitrary color space and computes a 1D histogram from it by
	 * dividing the channels into several regions, while the channels will be considered seperately from each other.
	 * @details This function divides the defined color space channels into several subspaces (depending on the set number of channel regions) and
	 * assigns each pixel in a given image to one of these subspaces, depending on the corresponding channel value. This results into a
	 * 1D histogram, that represents the colors in the image.
	 * The following color spaces are recognized:
	 *
	 *	- 0: RGB
	 *	- 1: rgb (channel normalized RGB)
	 *	- 2: R_IIG_IIB_II (across-component normalized RGB, see @cite yang_color_2010)
	 *	- 3: XYZ
	 *	- 4: xyY
	 *	- 5: X_IIY_IIZ_II (across-component normalized XYZ, see @cite yang_color_2010)
	 *	- 6: HSV
	 *	- 7: CIEL*a*b*
	 *	- 8: opponent color space
	 *	- 9: I_1I_2I_3
	 *	- 10: C_1C_2C_3
	 *	- 11: LSLM, see @cite yang_color_2010
	 *	- 12: CIE-LUV
	 *	- 13: l_1l_2l_3, (see @cite gevers_color-based_1999 for details)
	 *
	 * @param image: The image, for which the histogram should be computed.
	 * @param color_space: Parameter that indicates in which color space the given image is encoded (different color spaces have different
	 * ranges in the channels).
	 * @param histogram: The resulting 1D histogram.
	 * @param region_of_interest: Optional pointer parameter to a binary image (CV_8UC1) that shows with white pixels which pixels should
	 * be taken into account for the histogram computation.
	 * @param regions: The regions to which the pixels have been assigned during the procedure, depends on the defined color space. Optional
	 * parameter that can be used to get the regions to which the pixels have been assigned. The regions for each channel will be stored in the
	 * following way: For the channel a region is defined, the lower (pair.first) and the upper (pair.second) bound is stored, for the other
	 * channels simply a pair of (0,0) is stored to indicate that this channel is irrelevant for in this region.
	 * @param region_numbers: Optional pointer to a cv::Vec3i object that stores how many regions shall be used
	 * per channel.
	 */
	void calculateChannelWiseHistogram(const cv::Mat& image, const size_t& color_space, std::vector<float>& histogram,
									   const cv::Mat* region_of_interest=0, std::vector<std::pair<cv::Vec3f, cv::Vec3f> >* bin_regions=0,
									   const cv::Vec3i* region_numbers=0) const;

	/**
	 * @brief coutHistogram is a helper function that prints out a histogram at cout in a really simplye way.
	 * @details This function prints the given histogram in a really simple way to cout. It will look like:
	 *	|----|----o----|----|
	 *	********              : 0
	 *	****                  : 1
	 *  **************        : 2
	 *                        : 3
	 *	*****                 : 4
	 *	|----|----o----|----|
	 * The function assumes that the histogram is normalized (meaning that the sum of the entries equals 1).
	 * @param histogram: The histogram that shall be plotted.
	 * @param max_number_of_stars: Indicates how many stars represent 100%. Also controls how fine the
	 * resolution of the output will be.
	 */
	void coutHistogram(const std::vector<float>& histogram, const int max_number_of_stars=20);
};
