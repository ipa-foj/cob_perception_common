// header include
#include <cob_vision_utils/color_space_converter.hpp>

// std includes
#include <cmath>

/*
 * Constructor that sets the parameters of the implemented transformations.
 */
ColorSpaceConverter::ColorSpaceConverter(const uint& vertical_divisons, const uint& horizontal_divisons, const bool standardize_output_image)
{
	// ********** set the given parameters **********
	vertical_divisons_ = vertical_divisons;
	horizontal_divisons_ = horizontal_divisons;
}

/*
 * Function that goes through the pixels in the given patch and computes the mean and variance in each color channel.
 */
template<typename ValueClass>
void ColorSpaceConverter::calculatePatchStatistics(const cv::Mat& patch, std::vector<std::pair<double, double> >& channel_statistics)
{
	// ********** initialize the result vector **********
	channel_statistics = std::vector<std::pair<double, double> >(3);

	// ********** go through each pixel and calculate the mean for each channel **********
	double first_channel_mean=0.0, second_channel_mean=0.0, third_channel_mean=0.0;
	for(size_t y=0; y<patch.rows; ++y)
	{
		for(size_t x=0; x<patch.cols; ++x)
		{
			// ------- get the current pixel -------
			ValueClass v = patch.at<ValueClass>(y, x);

			// ------- update the mean values -------
			first_channel_mean += v[0];
			second_channel_mean += v[1];
			third_channel_mean += v[2];
		}
	}

	// ------- calculate the means -------
	double number_of_pixels = patch.rows*patch.cols;
	first_channel_mean /= number_of_pixels;
	channel_statistics[0].first = first_channel_mean;
	second_channel_mean /= number_of_pixels;
	channel_statistics[1].first = second_channel_mean;
	third_channel_mean /= number_of_pixels;
	channel_statistics[2].first = third_channel_mean;

	// ********** go through each pixel and calculate the variance for each channel **********
	double first_channel_var=0.0, second_channel_var=0.0, third_channel_var=0.0;
	for(size_t y=0; y<patch.rows; ++y)
	{
		for(size_t x=0; x<patch.cols; ++x)
		{
			// ------- get the current pixel -------
			ValueClass v = patch.at<ValueClass>(y, x);

			// ------- update the mean values -------
			first_channel_var += std::pow(v[0]-first_channel_mean, 2.0);
			second_channel_var += std::pow(v[1]-second_channel_mean, 2.0);
			third_channel_var += std::pow(v[2]-third_channel_mean, 2.0);
		}
	}

	// ------- calculate the variances -------
	channel_statistics[0].second = first_channel_var/number_of_pixels;
	channel_statistics[1].second = second_channel_var/number_of_pixels;
	channel_statistics[2].second = third_channel_var/number_of_pixels;
}

/*
 * API function that can be used to conveniently convert an RGB image to another color space.
 */
int ColorSpaceConverter::transformColorSpace(const cv::Mat& input, cv::Mat& output, const ColorSpaceConvertionType& type)
{
	// ********** call the function that converts the given image to the desired color space **********
	switch (type)
	{
		case ColorSpaceConvertionType::RGB2RGB:
		{
			// ------- just convert the given input image to CV_32FC3 format (and check for the correct input type) -------
			if(input.channels()!=3 || input.type()!=CV_8UC3)
			{
				// the given image is not in the correct format, return a failure indicator
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to RGB failed: Wrong input format!!" << std::endl;
				return -1;
			}
			input.convertTo(output, CV_32FC3);
			break;
		}
		case ColorSpaceConvertionType::RGB2rgb:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2rgb(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to rgb failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2RGBII:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2RGBII(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to R_IIG_IIB_II failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2XYZ:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2XYZ(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to XYZ failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2xyY:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2xyY(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to XYZ failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2XYZII:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2XYZII(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from R_IIG_IIB_II to rgb failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2HSV:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2HSV(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to HSV failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2Lab:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2Lab(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to Lab failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2Opponent:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2Opponent(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to opponent-space failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2I3:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2I3(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to I_1I_2I_3 failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2C3:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2C3(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to I_1I_2I_3 failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2LSLM:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2LSLM(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr  << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to LSLM failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2ZRG:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2ZRG(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to ZRG failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2Luv:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2Luv(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to Lab failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		case ColorSpaceConvertionType::RGB2l3:
		{
			// ------- try to convert the given image to the desired space, if this is not possible, return a failure -------
			try
			{
				RGB2l3(input, output);
			}
			catch(cv::Exception e)
			{
				std::cerr << std::endl << "Error [ColorSpaceConverter]: Converting from RGB to l3 failed: " << std::endl << e.what() << std::endl;
				return -1;
			}
			break;
		}
		default:
		{
			// ------- the given type wasn't recognized, simply return a failure indicator -------
			std::cerr << std::endl << "Error [ColorSpaceConverter]: Conversion type not recognized!!" << std::endl;
			return -1;
		}
	}

	// ********** if everything passes without problems, return a success indicator **********
	return 0;
}

/*
 * Function to convert an image from the RGB space to the rgb space.
 */
void ColorSpaceConverter::RGB2rgb(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2rgb conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2rgb conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			float sum = r + g + b;
			if(sum==0.0f)
				sum = 1.0f; // prevent division by 0
			output.at<cv::Vec3f>(y, x)[0] = (float(r)/sum);
			output.at<cv::Vec3f>(y, x)[1] = (float(g)/sum);
			output.at<cv::Vec3f>(y, x)[2] = (float(b)/sum);
		}
	}
}

/*
 * Function to convert an image from the RGB space to the RGBII space (across-color-component normalization, see
 * Color Space normalization, Yang et.al.).
 */
void ColorSpaceConverter::RGB2RGBII(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2RGBII conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2RGBII conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion (for details on this one see: Color Space normalization, Yang et.al.) -------
			output.at<cv::Vec3f>(y, x)[0] = (r);
			output.at<cv::Vec3f>(y, x)[1] = (-0.5774f*r + 0.7887f*g - 0.2113f*b);
			output.at<cv::Vec3f>(y, x)[2] = (-0.5774f*r - 0.2113f*g + 0.7887*b);
		}
	}
}

/*
 * Function that converts an image from the RGB space into the XYZ space.
 */
void ColorSpaceConverter::RGB2XYZ(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2XYZ conversion has wrong number of channels!");
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2XYZ conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** do the convertion using opencv **********
	cv::Mat input_clone;
	input.convertTo(input_clone, CV_32FC3);
	cv::cvtColor(input_clone, output, CV_RGB2XYZ);
}

/*
 * Function that converts an image from the RGB space into the xyY space.
 */
void ColorSpaceConverter::RGB2xyY(const cv::Mat& input, cv::Mat& output)
{
	// ********** use the above defined function to convert the image into the XYZ space **********
	RGB2XYZ(input, output);

	// ********** go through the pixels and calculate the xyY values **********
	for(size_t y=0; y<output.rows; ++y)
	{
		for(size_t x=0; x<output.cols; ++x)
		{
			// ------- get the current pixel values -------
			cv::Vec3f v = output.at<cv::Vec3f>(y, x);
			float X = v[0];
			float Y = v[1];
			float Z = v[2];

			// ------- normalize the color space -------
			float sum = X + Y + Z;
			if(sum==0.0f)
				sum = 1.0f; // prevent division by 0
			output.at<cv::Vec3f>(y, x)[0] = (X/sum);
			output.at<cv::Vec3f>(y, x)[1] = (Y/sum);
			output.at<cv::Vec3f>(y, x)[2] = (Y);
		}
	}
}

/*
 * Function to convert an image from the RGB space to the X_IIY_IZ_II space (across-color-component normalization, see
 * Color Space normalization, Yang et.al.).
 */
void ColorSpaceConverter::RGB2XYZII(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2XYZII conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2XYZII conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion (for details on this one see: Color Space normalization, Yang et.al.) -------
			output.at<cv::Vec3f>(y, x)[0] = (0.607f*r + 0.1740f*g + 0.2f*b);
			output.at<cv::Vec3f>(y, x)[1] = (-0.0901f*r + 0.3631f*g - 0.2730f*b);
			output.at<cv::Vec3f>(y, x)[2] = (-0.46f*r - 0.1986f*g + 0.6586*b);
		}
	}
}

/*
 * Function to convert an image from the RGB space to the HSV space.
 */
void ColorSpaceConverter::RGB2HSV(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2HSV conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2HSV conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** do the convertion using opencv **********
	cv::Mat input_clone;
	input.convertTo(input_clone, CV_32FC3);
	cv::cvtColor(input_clone, output, CV_RGB2HSV);
}

/*
 * Function to convert an image from the RGB space to the CIEL*a*b* space.
 */
void ColorSpaceConverter::RGB2Lab(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2Lab conversion has wrong number of channels!");
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2Lab conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** do the convertion using opencv **********
	cv::Mat input_clone;
	input.convertTo(input_clone, CV_32FC3);
	cv::cvtColor(input_clone, output, CV_RGB2Lab);
}

/*
 * Function to convert an image from the RGB space to the opponent-color-space.
 */
void ColorSpaceConverter::RGB2Opponent(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2Opponent conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2Opponent conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	float sqrt_of_two = std::sqrt(2.0f);
	float sqrt_of_three = std::sqrt(3.0f);
	float sqrt_of_six = std::sqrt(6.0f);
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			output.at<cv::Vec3f>(y, x)[0] = ((r - g)/sqrt_of_two);
			output.at<cv::Vec3f>(y, x)[1] = ((r + g - 2.0f*b)/sqrt_of_six);
			output.at<cv::Vec3f>(y, x)[2] = ((r + g + b)/sqrt_of_three);
		}
	}
}

/*
 * Function to convert an image from the RGB space to the I_1I_2I_3 space.
 */
void ColorSpaceConverter::RGB2I3(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2I3 conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2I3 conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	float one_third = 1.0f/3.0f;
	float one_half = 1.0f/2.0f;
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			output.at<cv::Vec3f>(y, x)[0] = (one_third*r + one_third*g + one_third*b);
			output.at<cv::Vec3f>(y, x)[1] = (one_half*r - one_half*b);
			output.at<cv::Vec3f>(y, x)[2] = (g - one_half*b - one_half*r);
		}
	}
}

/*
 * Function to convert an image from the RGB space to the C_1C_2C_3 space.
 */
void ColorSpaceConverter::RGB2C3(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2C3 conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2C3 conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			float c_one_input = float(r)/std::max(float(std::max(g, b)), 1e-6f); // prevent from dividing by 0 (arctan converges for large values)
			float c_two_input = float(g)/std::max(float(std::max(r, b)), 1e-6f); // prevent from dividing by 0
			float c_three_input = float(b)/std::max(float(std::max(r, g)), 1e-6f); // prevent from dividing by 0
			output.at<cv::Vec3f>(y, x)[0] = (std::atan(c_one_input));
			output.at<cv::Vec3f>(y, x)[1] = (std::atan(c_two_input));
			output.at<cv::Vec3f>(y, x)[2] = (std::atan(c_three_input));
		}
	}
}

/*
 * Function to convert an image from the RGB space size_to the LSLM space.
 */
void ColorSpaceConverter::RGB2LSLM(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2LSLM conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2LSLM conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			output.at<cv::Vec3f>(y, x)[0] = (0.209f*(r-0.5f) + 0.715f*(g-0.5f) + 0.076f*(b-0.5f));
			output.at<cv::Vec3f>(y, x)[1] = (0.209f*(r-0.5f) + 0.715f*(g-0.5f) - 0.924f*(b-0.5f));
			output.at<cv::Vec3f>(y, x)[2] = (3.148f*(r-0.5f) - 2.799f*(g-0.5f) - 0.349f*(b-0.5f));
		}
	}
}

/*
 * Function to convert an image from the RGB space to the CIE-LUV space.
 */
void ColorSpaceConverter::RGB2Luv(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2Lab conversion has wrong number of channels!");
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2Lab conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** do the convertion using opencv **********
	cv::Mat input_clone;
	input.convertTo(input_clone, CV_32FC3);
	input_clone *= 1.0/255.0; // the input image in CV_32FC3 format is assumed to be in [0, 1]
	cv::cvtColor(input_clone, output, CV_RGB2Luv);
}

/*
 * Function to convert an image from the RGB space to the l_1l_2l_3 space.
 */
void ColorSpaceConverter::RGB2l3(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2l3 conversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2l3 conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			float r_g_square = std::pow(r - g, 2.0);
			float r_b_square = std::pow(r - b, 2.0);
			float g_b_square = std::pow(g - b, 2.0);
			float sum = r_g_square + r_b_square + g_b_square;
			if(sum==0.0f)
				sum = 1.0f; // prevent division by 0
			output.at<cv::Vec3f>(y, x)[0] = (r_g_square/sum);
			output.at<cv::Vec3f>(y, x)[1] = (r_b_square/sum);
			output.at<cv::Vec3f>(y, x)[2] = (g_b_square/sum);
		}
	}
}

/*
 * Function to convert an image from the RGB space size_to the ZRG space.
 */
void ColorSpaceConverter::RGB2ZRG(const cv::Mat& input, cv::Mat& output)
{
	// ********** make sure the input image has the correct number of channels and the right format **********
	if(input.channels()!=3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2ZRGconversion has wrong number of channels! Expected 3, got " + std::to_string(input.channels()));
	}
	else if(input.type()!=CV_8UC3)
	{
		CV_Error(CV_StsBadArg, "Input image for RGB2ZRG conversion has wrong type! Expected 16, got " + std::to_string(input.type()));
	}

	// ********** resize the output to the correct format **********
	output = cv::Mat(input.rows, input.cols, CV_32FC3);

	// ********** convert the color space **********
	for(size_t y = 0; y<input.rows; ++y)
	{
		for(size_t x = 0; x<input.cols; ++x)
		{
			// ------- read out the given rgb values -------
			cv::Vec3b v = input.at<cv::Vec3b>(y, x);
			uchar& r = v[0];
			uchar& g = v[1];
			uchar& b = v[2];

			// ------- do the conversion -------
			output.at<cv::Vec3f>(y, x)[0] = (0.02018311f*r + 0.12955342f*g + 0.93918125f*b);
			output.at<cv::Vec3f>(y, x)[1] = (r);
			output.at<cv::Vec3f>(y, x)[2] = (g);
		}
	}
}
