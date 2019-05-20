// header include
#include <cob_vision_utils/histogram_computer.hpp>

// OpenMP
#include <omp.h>

/*
 * Function that normalizes the given histogram by doing a vector normaliztaion, using the L1 norm.
 */
void ColorHistogramComputer::normalizeHistogram(std::vector<float>& histogram) const
{
	// ********** calculate the sum **********
	float magnitude = 0.0;
	for(const float& e : histogram)
		magnitude += e;

	// ********** modify each entry **********
	for(float& e : histogram)
		e /= magnitude;
}

/*
 * Setter function for the number of threads this computer is allowed to open.
 */
void ColorHistogramComputer::setNumberOfThreads(const size_t& number_of_threads)
{
	// ********** set the number of threads **********
	number_of_threads_ = number_of_threads;
}

/*
 * Function to calculate the bin regions for the specified color space.
 */
void ColorHistogramComputer::getRegionsForColorSpace(std::vector<std::pair<cv::Vec3f, cv::Vec3f> >& regions,
													 const size_t& color_space, const size_t& c1_regions,
													 const size_t& c2_regions, const size_t& c3_regions) const
{
	// ********** get the regions for the wanted color space **********
	float c_1_delta = (color_space_channel_ranges_[color_space].second[0]-color_space_channel_ranges_[color_space].first[0])/c1_regions;
	float c_2_delta = (color_space_channel_ranges_[color_space].second[1]-color_space_channel_ranges_[color_space].first[1])/c2_regions;
	float c_3_delta = (color_space_channel_ranges_[color_space].second[2]-color_space_channel_ranges_[color_space].first[2])/c3_regions;
	for(float c_1_lb=color_space_channel_ranges_[color_space].first[0]; c_1_lb<=color_space_channel_ranges_[color_space].second[0]-c_1_delta+1e-2;
		c_1_lb+=c_1_delta)
	{
		// catch the float precision when checking for the upper border
		for(float c_2_lb=color_space_channel_ranges_[color_space].first[1]; c_2_lb<=color_space_channel_ranges_[color_space].second[1]-c_2_delta+1e-2;
			c_2_lb+=c_2_delta)
		{
			for(float c_3_lb=color_space_channel_ranges_[color_space].first[2]; c_3_lb<=color_space_channel_ranges_[color_space].second[2]-c_3_delta+1e-2;
				c_3_lb+=c_3_delta)
			{
				regions.push_back(std::make_pair<cv::Vec3f>(cv::Vec3f(c_1_lb, c_2_lb, c_3_lb),
															cv::Vec3f(c_1_lb+c_1_delta, c_2_lb+c_2_delta, c_3_lb+c_3_delta)));
			}
		}
	}
}

/*
 * Constructor that sets up the ranges in the different channels for each color space.
 */
ColorHistogramComputer::ColorHistogramComputer(const size_t& number_of_threads)
{
	// ********** set the number of allowed parallel threads **********
	number_of_threads_ = number_of_threads;

	// ********** define the min/max values in the channels for the different color spaces **********
	// RGB
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(255.0, 255.0, 255.0)));

	// rgb
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(1.0, 1.0, 1.0)));

	// R_IIG_IIB_II
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, -201.12, -201.12),
																			   cv::Vec3f(255.0, 201.12, 201.12)));

	// XYZ
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(242.37, 255.0, 277.64)));

	// xyY
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(1.0, 1.0, 255.0)));

	// X_IIY_IIZ_II
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, -92.6, -167.95),
																			   cv::Vec3f(250.2, 92.6, 167.95)));

	// HSV
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(360.0, 1.0, 255.0)));

	// Lab
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, -86.2, -107.9),
																			   cv::Vec3f(100.1, 98.3, 94.5)));

	// opponent color space
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(-180.35, -208.21, 0.0),
																			   cv::Vec3f(180.35, 208.21, 441.7)));

	// I1I2I3
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, -127.5, -255.0),
																			   cv::Vec3f(255.0, 127.5, 255.0)));

	// c1c2c3
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(1.6, 1.6, 1.6)));

	// LSLM
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(-0.5, -235.63, -802.75),
																			   cv::Vec3f(254.5, 235.63, 802.75)));

	// Luv
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, -83.1, -134.1),
																			   cv::Vec3f(100.1, 175.1, 107.4)));

	// l1l2l3
	color_space_channel_ranges_.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0.0, 0.0, 0.0),
																			   cv::Vec3f(0.7, 0.7, 0.7)));
}

/*
 * Function that goes through every pixel in the given image and assigns it to a bin, that is defined by the set number of channel divisions and
 * the defined color space.
 * REMARK: The given image is assumed to be in the CV_32FC3 format.
 */
void ColorHistogramComputer::calculateHistogram(const cv::Mat& image, const size_t& color_space, std::vector<float>& histogram, const cv::Mat* region_of_interest,
												std::vector<std::pair<cv::Vec3f, cv::Vec3f> >* bin_regions, const cv::Vec3i* region_numbers) const
{
	// ********** check if the provided color space is available (in the defined range of spaces) **********
	if(color_space<0 || color_space>13)
	{
		// ------- the provided color space is not available, print an error and return -------
		std::cerr << "!!! Warning [ColorHistogramComputer]: The given color space " << color_space << " isn't existing for the histogram computation. Aborting... !!!" << std::endl;
		return;
	}

	// ********** check if a different number of regions shall be used than the defined ones **********
	size_t c1_regions, c2_regions, c3_regions;
	if(region_numbers!=0)
	{
		c1_regions = region_numbers->operator [](0);
		c2_regions = region_numbers->operator [](1);
		c3_regions = region_numbers->operator [](2);
	}
	else
	{
		c1_regions = channelmerged_channel_one_region_number_;
		c2_regions = channelmerged_channel_two_region_number_;
		c3_regions = channelmerged_channel_three_region_number_;
	}

	// ********** get the different ranges, depending on the defined number of divisions of the channels **********
	std::vector<std::pair<cv::Vec3f, cv::Vec3f> > regions;
	getRegionsForColorSpace(regions, color_space, c1_regions, c2_regions, c3_regions);

	// ********** go through the pixels and check to which color space it belongs **********
	histogram.resize(regions.size(), 0.0); // bring the histogram vector to the correct size and initialize each bin as empty
	omp_set_dynamic(0); // disable dynamic teams
	omp_set_num_threads(number_of_threads_); // use the maximal allowed number of threads

	for(size_t y=0; y<image.rows; ++y)
	{
#pragma omp parallel for ordered // let OpenMP open up threads for each row
		for(size_t x=0; x<image.cols; ++x)
		{
			// ------- if a region of interest was provided, check if the current pixel is inside it -------
			if(region_of_interest!=0)
				if(region_of_interest->at<uchar>(y, x)<250)
					continue;

			// ------- find the bin that this pixel belongs to -------
			const cv::Vec3f v = image.at<cv::Vec3f>(y, x);
			auto bin_iterator = std::find_if(regions.begin(), regions.end(), [v](const std::pair<cv::Vec3f, cv::Vec3f>& a){return (v>=a.first && v<=a.second);});

			// ------- increase the number of pixels in the corresponding bin -------
			if(bin_iterator!=regions.end())
			{
				size_t bin_index = std::distance(regions.begin(), bin_iterator);
				std::lock_guard<std::mutex> lock(access_histogram_bin_mutex_);
				++histogram[bin_index];
			}
			else
			{
				// ------- the given pixel wasn't in the correct range, print an error -------
				std::lock_guard<std::mutex> lock(access_histogram_bin_mutex_);
				std::cerr << "!!! Warning [ColorHistogramComputer]: The pixel (" << v[0] << ", " << v[1] << ", " << v[2] << ") is not in the defined range of the color space " << color_space << " !!!" << std::endl;
			}
		}
	}

	// ********** normalize the created histogram **********
	normalizeHistogram(histogram);

	// ********** if wanted return the created regions **********
	if(bin_regions!=0)
		*bin_regions = std::move(regions);
}

/*
 * Function that goes through the pixels and assigns the channels of each pixel to one bin of a histogram, corresponding to the channel of the color
 * space. After every pixel (that is inside the provided region of interest) has been assigned to a bin, the 3 channel histograms will be concentrated
 * in one histogram that represents the color in the whole image. This resulting histogram is then normalized, s.t. the bin-values represent the
 * percentages of each corresponding channel value in the given image.
 * REMARK: The given image is assumed to be in the CV_32FC3 format.
 */
void ColorHistogramComputer::calculateChannelWiseHistogram(const cv::Mat& image, const size_t& color_space, std::vector<float>& histogram,
														   const cv::Mat* region_of_interest,
														   std::vector<std::pair<cv::Vec3f, cv::Vec3f> >* bin_regions, const cv::Vec3i* region_numbers) const
{
	// ********** check if the provided color space is available (in the defined range of spaces) **********
	if(color_space<0 || color_space>13)
	{
		// ------- the provided color space is not available, print an error and return -------
		std::cerr << "!!! Warning [ColorHistogramComputer]: The given color space " << color_space << " isn't existing for the histogram computation. Aborting... !!!" << std::endl;
		return;
	}

	// ********** check if a different number of regions shall be used than the defined ones **********
	size_t c1_regions, c2_regions, c3_regions;
	if(region_numbers!=0)
	{
		c1_regions = region_numbers->operator [](0);
		c2_regions = region_numbers->operator [](1);
		c3_regions = region_numbers->operator [](2);
	}
	else
	{
		c1_regions = channelmerged_channel_one_region_number_;
		c2_regions = channelmerged_channel_two_region_number_;
		c3_regions = channelmerged_channel_three_region_number_;
	}

	// ********** get the different ranges, depending on the defined number of divisions of the channels **********
	float c_1_delta = (color_space_channel_ranges_[color_space].second[0]-color_space_channel_ranges_[color_space].first[0])/c1_regions;
	float c_2_delta = (color_space_channel_ranges_[color_space].second[1]-color_space_channel_ranges_[color_space].first[1])/c2_regions;
	float c_3_delta = (color_space_channel_ranges_[color_space].second[2]-color_space_channel_ranges_[color_space].first[2])/c3_regions;
	std::vector<std::pair<cv::Vec3f, cv::Vec3f> > channel_one_regions, channel_two_regions, channel_three_regions;

	// catch the float precision when checking for the upper border
	for(float c_1_lb=color_space_channel_ranges_[color_space].first[0]; c_1_lb<=color_space_channel_ranges_[color_space].second[0]-c_1_delta+1e-2;
		c_1_lb+=c_1_delta)
	{
		channel_one_regions.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(c_1_lb, 0, 0),
																		   cv::Vec3f(c_1_lb+c_1_delta, 0, 0)));
	}
	for(float c_2_lb=color_space_channel_ranges_[color_space].first[1]; c_2_lb<=color_space_channel_ranges_[color_space].second[1]-c_2_delta+1e-2;
		c_2_lb+=c_2_delta)
	{
		channel_two_regions.push_back(std::make_pair<cv::Vec3f, cv::Vec3f>(cv::Vec3f(0, c_2_lb, 0),
																		   cv::Vec3f(0, c_2_lb+c_2_delta, 0)));
	}
	for(float c_3_lb=color_space_channel_ranges_[color_space].first[2]; c_3_lb<=color_space_channel_ranges_[color_space].second[2]-c_3_delta+1e-2;
		c_3_lb+=c_3_delta)
	{
		channel_three_regions.push_back(std::make_pair<cv::Vec3f>(cv::Vec3f(0, 0, c_3_lb),
																  cv::Vec3f(0, 0, c_3_lb+c_3_delta)));
	}

	// ********** go through the pixels and assign each channel value to one of the bins in the corresponding histogram **********
	std::vector<double> channel_one_histogram(channel_one_regions.size(), 0.0), channel_two_histogram(channel_two_regions.size(), 0.0),
			channel_three_histogram(channel_three_regions.size(), 0.0);
	omp_set_dynamic(0); // disable dynamic teams
	omp_set_num_threads(number_of_threads_); // use the maximal allowed number of threads

	for(size_t y=0; y<image.rows; ++y)
	{
#pragma omp parallel for ordered // let OpenMP open up threads for each row
		for(size_t x=0; x<image.cols; ++x)
		{
			// if a region of interest was provided, check if the current pixel is inside it
			if(region_of_interest!=0)
				if(region_of_interest->at<uchar>(y, x)<250)
					continue;

			// access the current pixel
			const cv::Vec3f v = image.at<cv::Vec3f>(y, x);

			// check the regions of each channel histogram and find the bin that this pixel-channel-value belongs to
			auto channel_one_position = std::find_if(channel_one_regions.begin(), channel_one_regions.end(), [v](const std::pair<cv::Vec3f, cv::Vec3f>& r)
			{return (v[0]>r.first[0] || std::abs(v[0]-r.first[0])<=1e-7) && (v[0]<r.second[0] || std::abs(v[0]-r.second[0])<=1e-7);});
			if(channel_one_position!=channel_one_regions.end())
			{
				size_t bin_index = std::distance(channel_one_regions.begin(), channel_one_position);
				std::lock_guard<std::mutex> lock(access_histogram_bin_mutex_);
				++channel_one_histogram[bin_index];
			}
			else
			{
				std::cerr << "!!! Warning [ColorHistogramComputer]: Channel value " << v[0] << " not in the defined range of color space " << color_space << std::endl;
			}
			auto channel_two_position = std::find_if(channel_two_regions.begin(), channel_two_regions.end(), [v](const std::pair<cv::Vec3f, cv::Vec3f>& r)
			{return (v[1]>r.first[1] || std::abs(v[1]-r.first[1])<=1e-7) && (v[1]<r.second[1] || std::abs(v[1]-r.second[1])<=1e-7);});
			if(channel_two_position!=channel_two_regions.end())
			{
				size_t bin_index = std::distance(channel_two_regions.begin(), channel_two_position);
				std::lock_guard<std::mutex> lock(access_histogram_bin_mutex_);
				++channel_two_histogram[bin_index];
			}
			else
			{
				std::cerr << "!!! Warning [ColorHistogramComputer]: Channel value " << v[1] << " not in the defined range of color space " << color_space << std::endl;
			}
			auto channel_three_position = std::find_if(channel_three_regions.begin(), channel_three_regions.end(), [v](const std::pair<cv::Vec3f, cv::Vec3f>& r)
			{return (v[2]>r.first[2] || std::abs(v[2]-r.first[2])<=1e-7) && (v[2]<r.second[2] || std::abs(v[2]-r.second[2])<=1e-7);});
			if(channel_three_position!=channel_three_regions.end())
			{
				size_t bin_index = std::distance(channel_three_regions.begin(), channel_three_position);
				std::lock_guard<std::mutex> lock(access_histogram_bin_mutex_);
				++channel_three_histogram[bin_index];
			}
			else
			{
				std::cerr << "!!! Warning [ColorHistogramComputer]: Channel value " << v[2] << " not in the defined range of color space " << color_space << std::endl;
			}
		}
	}

	// ********** concentrate the resulting channel histograms into one overall histogram **********
	histogram.resize(number_of_separate_regions_);
	size_t bin_index_counter=0;
	for(double& val : channel_one_histogram)
		histogram[bin_index_counter++] = std::move(val);
	for(double& val : channel_two_histogram)
		histogram[bin_index_counter++] = std::move(val);
	for(double& val : channel_three_histogram)
		histogram[bin_index_counter++] = std::move(val);

	// ********** normalize the resulting histogram **********
	normalizeHistogram(histogram);

	// ********** if wanted store the regions **********
	if(bin_regions!=0)
	{
		bin_regions->insert(bin_regions->end(), channel_one_regions.begin(), channel_one_regions.end());
		bin_regions->insert(bin_regions->end(), channel_two_regions.begin(), channel_two_regions.end());
		bin_regions->insert(bin_regions->end(), channel_three_regions.begin(), channel_three_regions.end());
	}
}

/*
 * Function to plot the given histogram in a simple way to cout.
 */
void ColorHistogramComputer::coutHistogram(const std::vector<float>& histogram, const int max_number_of_stars)
{
	// ********** go through the elements of the histogram and plot the histogram **********
//	std::cout << "---|----|----o----|----|" << std::endl;
	std::cout << "|";
	for(int counter=1; counter<max_number_of_stars-1; ++counter)
	{
		if(counter%(max_number_of_stars/4)==0)
		{
			if(counter==0.5*max_number_of_stars)
				std::cout << "o";
			else
				std::cout << "|";
		}
		else
			std::cout << "-";
	}
	std::cout << "|" << std::endl;
	size_t counter=0;
	for(const float entry : histogram)
	{
		int number_of_stars = (float) max_number_of_stars*entry;
		for(int i=1; i<=number_of_stars; ++i)
			std::cout << "*";
		for(int i=number_of_stars; i<=max_number_of_stars; ++i)
			std::cout << " ";
		std::cout << ": " << counter++ << std::endl;
	}
	std::cout << "|";
	for(int counter=1; counter<max_number_of_stars-1; ++counter)
	{
		if(counter%(max_number_of_stars/4)==0)
		{
			if(counter==0.5*max_number_of_stars)
				std::cout << "o";
			else
				std::cout << "|";
		}
		else
			std::cout << "-";
	}
	std::cout << "|" << std::endl;
}
