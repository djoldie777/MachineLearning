#define _CRT_SECURE_NO_WARNINGS

#include "FormData.h"
#include <fstream>

#define TRAINING_SAMPLES 3900
#define ATTRIBUTES 400
#define TEST_SAMPLES 1300
#define CLASSES 26

void getData(int **pixel_matrix, cv::Mat &data, cv::Mat &classes, int samples_count)
{
	for (int row = 0; row < samples_count; row++)
	{
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			if (col < ATTRIBUTES)
				data.at<float>(row, col) = (float)pixel_matrix[row][col];
			else if (col == ATTRIBUTES)
				classes.at<float>(row, pixel_matrix[row][col] - 1) = 1.0f;
		}
	}
}

void getData_(int **pixel_matrix, cv::Mat &data, cv::Mat &classes, int samples_count)
{
	for (int row = 0; row < samples_count; row++)
	{
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			if (col < ATTRIBUTES)
				data.at<float>(row, col) = (float)pixel_matrix[row][col];
			else if (col == ATTRIBUTES)
				classes.at<float>(row, 0) = (float)(pixel_matrix[row][col] - 1);
		}
	}
}

void multilayerPerceptron(int **training_matrix, int **test_matrix)
{
	// матрица, содержащая образцы для обучения
	cv::Mat training_set(TRAINING_SAMPLES, ATTRIBUTES, CV_32F);
	// матрица, содержащая метки каждого образца для обучения
	cv::Mat training_set_classes(TRAINING_SAMPLES, CLASSES, CV_32F);
	// матрица, содержащая образцы для теста
	cv::Mat test_set(TEST_SAMPLES, ATTRIBUTES, CV_32F);
	// матрица, содержащая метки каждого образца для теста
	cv::Mat test_set_classes(TEST_SAMPLES, CLASSES, CV_32F);
	// матрица, содержащая веса каждого класса
	cv::Mat classification_result(1, CLASSES, CV_32F);

	getData(training_matrix, training_set, training_set_classes, TRAINING_SAMPLES);
	getData(test_matrix, test_set, test_set_classes, TEST_SAMPLES);

	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES;
	layers.at<int>(1, 0) = 40;
	layers.at<int>(2, 0) = CLASSES;

	CvANN_MLP network(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);
	CvANN_MLP_TrainParams params(cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);

	cout << "_______________________________________\n" << endl;
	cout << "Using Multilayer Perceptron training..." << endl;
	cout << "Training iterations: " << network.train(training_set, training_set_classes, cv::Mat(), cv::Mat(), params) << endl << endl;

	cv::Mat test_sample;
	int correct = 0, wrong = 0;

	for (int i = 0; i < TEST_SAMPLES; i++)
	{
		test_sample = test_set.row(i);
		network.predict(test_sample, classification_result);

		// поиск класса с максимальными весами
		float max = classification_result.at<float>(0, 0);
		int maxIndex = 0;

		for (int j = 1; j < CLASSES; j++)
		{
			float tmp = classification_result.at<float>(0, j);

			if (tmp > max)
			{
				max = tmp;
				maxIndex = j;
			}
		}

		if (test_set_classes.at<float>(i, maxIndex) == 1.0f)
			correct++;
		else
			wrong++;
	}

	cout << "Correct classification: " << (double)correct / TEST_SAMPLES * 100 << "%" << endl;
	cout << "_______________________________________\n" << endl;
}

void kNearestNeighbors(int **training_matrix, int **test_matrix)
{
	cv::Mat training_set(TRAINING_SAMPLES, ATTRIBUTES, CV_32F);
	cv::Mat training_set_classes(TRAINING_SAMPLES, 1, CV_32F);
	cv::Mat test_set(TEST_SAMPLES, ATTRIBUTES, CV_32F);
	cv::Mat test_set_classes(TEST_SAMPLES, 1, CV_32F);
	cv::Mat classification_result(TEST_SAMPLES, 1, CV_32F);

	getData_(training_matrix, training_set, training_set_classes, TRAINING_SAMPLES);
	getData_(test_matrix, test_set, test_set_classes, TEST_SAMPLES);

	cout << "Using K Nearest Neighbors training...\n" << endl;
	CvKNearest knn(training_set, training_set_classes);

	cv::Mat test_sample;
	int correct = 0, wrong = 0;

	for (int i = 0; i < TEST_SAMPLES; i++)
	{
		test_sample = test_set.row(i);
		classification_result.at<float>(i, 0) = knn.find_nearest(test_sample, 7);

		if (classification_result.at<float>(i, 0) == test_set_classes.at<float>(i, 0))
			correct++;
		else
			wrong++;
	}

	cout << "Correct classification: " << (double)correct / TEST_SAMPLES * 100 << "%" << endl;
	cout << "_______________________________________\n" << endl;
}

void decisionTree(int **training_matrix, int **test_matrix)
{
	cv::Mat training_set(TRAINING_SAMPLES, ATTRIBUTES, CV_32F);
	cv::Mat training_set_classes(TRAINING_SAMPLES, 1, CV_32F);
	cv::Mat test_set(TEST_SAMPLES, ATTRIBUTES, CV_32F);
	cv::Mat test_set_classes(TEST_SAMPLES, 1, CV_32F);
	cv::Mat classification_result(TEST_SAMPLES, 1, CV_32F);

	getData_(training_matrix, training_set, training_set_classes, TRAINING_SAMPLES);
	getData_(test_matrix, test_set, test_set_classes, TEST_SAMPLES);

	CvDTree dtree;
	cv::Mat var_type(ATTRIBUTES + 1, 1, CV_8U);
	var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL));
	var_type.at<uchar>(ATTRIBUTES, 0) = CV_VAR_CATEGORICAL;

	float *priors = NULL;
	CvDTreeParams params = CvDTreeParams(20, 2, 0, false, 10, 10, false, false, priors);

	cout << "Using Decision Tree training...\n" << endl;
	dtree.train(training_set, CV_ROW_SAMPLE, training_set_classes, cv::Mat(), cv::Mat(), var_type, cv::Mat(), params);

	cv::Mat test_sample;
	int correct = 0, wrong = 0;

	for (int i = 0; i < TEST_SAMPLES; i++)
	{
		test_sample = test_set.row(i);
		classification_result.at<float>(i, 0) = (float)dtree.predict(test_sample)->value;

		if (classification_result.at<float>(i, 0) == test_set_classes.at<float>(i, 0))
			correct++;
		else
			wrong++;
	}

	cout << "Correct classification: " << (double)correct / TEST_SAMPLES * 100 << "%" << endl;
	cout << "_______________________________________\n" << endl;
}

void releaseMatrixMemory(int **matrix, int samples_count)
{
	for (int i = 0; i < samples_count; i++)
		delete[] matrix[i];

	delete[] matrix;
}

void createFile(int **matrix, string outputfile, int samples_count)
{
	fstream file(outputfile, ios::out);

	for (int i = 0; i < samples_count; i++)
	{
		for (int j = 0; j <= ATTRIBUTES; j++)
		{
			if (j < ATTRIBUTES)
				file << matrix[i][j] << ",";
			else
				file << matrix[i][j] << "\n";
		}
	}

	file.close();
}

bool matrixComparing(char *filename1, char *filename2, int total_samples)
{
	float value1, value2;
	FILE *file1 = fopen(filename1, "r");
	FILE *file2 = fopen(filename2, "r");

	for (int row = 0; row < total_samples; row++)
	{
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			if (col < ATTRIBUTES)
			{
				fscanf(file1, "%f,", &value1);
				fscanf(file2, "%f,", &value2);

				if (value1 != value2)
					return false;
			}
			else if (col == ATTRIBUTES)
			{
				fscanf(file1, "%f", &value1);
				fscanf(file2, "%f", &value2);

				if (value1 != value2)
					return false;
			}
		}
	}

	fclose(file1);
	fclose(file2);

	return true;
}

int main()
{
	FormData fd("Train\\");

	cout << "_______________________________________\n" << endl;
	cout << "Forming the training set..." << endl;
	int **training_matrix = fd.getPixelMatrix(26, 150);
	fd.setPath("Test\\");
	cout << "Forming the test set..." << endl;
	int **test_matrix = fd.getPixelMatrix(26, 50);

	multilayerPerceptron(training_matrix, test_matrix);
	kNearestNeighbors(training_matrix, test_matrix);
	decisionTree(training_matrix, test_matrix);

	releaseMatrixMemory(training_matrix, TRAINING_SAMPLES);
	releaseMatrixMemory(test_matrix, TEST_SAMPLES);

	system("pause");
	return 0;
}