#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <fftw3.h>

using namespace std;
const int Directions = 180;
const double Pi = 3.14159265359;
const int N = 128;
double attenuation[Directions][2 * N];
double reconstructed[2 * N][2 * N];

void inputImage() {
    cv::Mat inputImage = cv::imread("./images/Sinogram_180d_16bit.png", cv::IMREAD_UNCHANGED);
    if (inputImage.empty()) {
        cerr << "Failed to load input image!" << endl;
        exit(1);
    }

    if (inputImage.rows != Directions || inputImage.cols != 2 * N) {
        cerr << "Input image dimensions do not match expected size!" << endl;
        exit(1);
    }

    for (int i = 0; i < Directions; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            attenuation[i][j] = static_cast<double>(inputImage.at<uint16_t>(i, j));
        }
    }
}

void outputImage(double * data, const string& filename) {
    cv::Mat outputImage(2 * N, 2 * N, CV_16U);
    for (int i = 0; i < 2 * N; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            outputImage.at<uint16_t>(i, j) = static_cast<uint16_t>(data[i * 2 * N + j]);
        }
    }
    if (!cv::imwrite(filename, outputImage)) {
        cerr << "Failed to write output image!" << endl;
        exit(1);
    }
}

int main() {
    inputImage();
    for (int i = -N; i < N; ++i) {
        for (int j = -N; j < N; ++j) {
            double x = static_cast<double>(i) / N;
            double y = static_cast<double>(j) / N;
            double sum = 0.0;
            for (int d = 0; d < Directions; ++d) {
                double alpha = Pi * d / Directions;
                double x_prime = x * cos(alpha) + y * sin(alpha);
                double y_prime = -x * sin(alpha) + y * cos(alpha);
                double t = N * (y_prime + 1.0);
                if (t >= 0 && t < 2 * N) {
                    int intt = static_cast<int>(t);
                    sum += attenuation[d][intt] * (intt + 1 - t) + attenuation[d][intt + 1] * (t - intt);
                }
            }
            reconstructed[i + N][j + N] = sum / Directions;
        }
    }
    outputImage(reinterpret_cast<double*>(reconstructed), "./images/reconstructed_180d_BP.png");
    return 0;
}