#pragma once
#ifndef FORMDATA_H
#define FORMDATA_H

#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>

using namespace std;

class FormData {
private:
	string path;

	void imageToVector(cv::Mat &img, int *pixelarray)
	{
		int i = 0;

		for (int x = 0; x < 20; x++)
		{
			for (int y = 0; y < 20; y++)
			{
				pixelarray[i] = (img.at<uchar>(x, y) == 255) ? 1 : 0;
				i++;
			}
		}
	}

public:
	FormData(string path)
	{
		this->path = path;
	}

	void setPath(string path)
	{
		this->path = path;
	}

	int** getPixelMatrix(int samples, int images)
	{
		int **res = new int*[samples * images];

		for (int img = 1; img <= images; img++)
		{
			for (int sample = 1; sample <= samples; sample++)
			{
				string image_path = path + "Sample" + to_string(sample) + "\\img" + to_string(img) + ".png";
				cv::Mat image = cv::imread(image_path);

				cv::Mat new_image;
				cv::cvtColor(image, new_image, CV_RGB2GRAY);
				cv::threshold(new_image, new_image, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

				int arr[400];
				imageToVector(new_image, arr);

				int index = (img - 1) * samples + (sample - 1);

				res[index] = new int[401];

				for (int i = 0; i < 400; i++)
					res[index][i] = arr[i];

				res[index][400] = sample;
			}
		}

		return res;
	}
};

#endif // FORMDATA_H