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
const int GrayLevel = 5000;
double attenuation[Directions][2 * N];
double reconstructed[2 * N][2 * N];

void inputImage() {
    cv::Mat inputImage = cv::imread("./images/Sinogram_180d_8bit.png", cv::IMREAD_UNCHANGED);
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
            attenuation[i][j] = static_cast<double>(inputImage.at<uint8_t>(i, j));
        }
    }
}

void outputImage(double * data, const string& filename, bool amplified = false) {
    double maximum = 0.0;
    for (int i = 0; i < 2 * N; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            if (data[i * 2 * N + j] > maximum) {
                maximum = data[i * 2 * N + j];
            }
        }
    }
    if (amplified) {
        vector<int> histogram(GrayLevel, 0);
        for (int i = 0; i < 2 * N; ++i) {
            for (int j = 0; j < 2 * N; ++j) {
                int value = static_cast<int>(data[i * 2 * N + j] / maximum * GrayLevel);
                if (value >= 0 && value < GrayLevel) {
                    histogram[value]++;
                }
            }
        }
        int peak_value = 0, peak_value_id = 0;
        for (int i = 0; i < GrayLevel; ++i) {
            if (histogram[i] > peak_value) {
                peak_value = histogram[i];
                peak_value_id = i;
            }
        }

        int L = peak_value_id, R = peak_value_id;
        for (; L > 0; --L) {
            if (histogram[L] < 0.01 * peak_value) {
                break;
            }
        }
        for (; R < GrayLevel; ++R) {
            if (histogram[R] < 0.01 * peak_value) {
                break;
            }
        }
        cv::Mat outputImage(2 * N, 2 * N, CV_8UC1);
        for (int i = 0; i < 2 * N; ++i) {
            for (int j = 0; j < 2 * N; ++j) {
                int value = static_cast<int>((data[i * 2 * N + j] / maximum * GrayLevel - L) / (R - L) * 255.0);
                value = std::max(0, std::min(255, value)); // Clamp to [0, 255]
                outputImage.at<uint8_t>(i, j) = static_cast<uint8_t>(value);
            }
        }
        cv::imwrite(filename, outputImage);
    }
    else {
        cv::Mat outputImage(2 * N, 2 * N, CV_8UC1);
        for (int i = 0; i < 2 * N; ++i) {
            for (int j = 0; j < 2 * N; ++j) {
                outputImage.at<uint8_t>(i, j) = static_cast<uint8_t>(data[i * 2 * N + j] / maximum * 255);
            }
        }
        cv::imwrite(filename, outputImage);
    }
}

void applyFilter(double* data, int size) {
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    for (int i = 0; i < size; ++i) {
        in[i][0] = data[i];
        in[i][1] = 0.0;
    }
    fftw_plan forwardPlan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(forwardPlan);
    for (int i = 0; i < size; ++i) {
        double filter = 2.0 * abs((i <= size / 2) ? i : size - i) / size;
        out[i][0] *= filter;
        out[i][1] *= filter;
    }
    fftw_plan backwardPlan = fftw_plan_dft_1d(size, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(backwardPlan);
    for (int i = 0; i < size; ++i) {
        data[i] = in[i][0] / size;
    }
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(backwardPlan);
    fftw_free(in);
    fftw_free(out);
}

int main() {
    inputImage();
    for (int i = 0; i < Directions; ++i) {
        applyFilter(attenuation[i], 2 * N);
    }
    for (int i = -N; i < N; ++i) {
        for (int j = -N; j < N; ++j) {
            double x = static_cast<double>(i) / N;
            double y = static_cast<double>(j) / N;
            double sum = 0.0;
            for (int d = 0; d < Directions; ++d) {
                double alpha = Pi * d / Directions;
                double x_prime = x * cos(alpha) + y * sin(alpha);
                double y_prime = -x * sin(alpha) + y * cos(alpha);
                // int t = static_cast<int>(N * (y_prime + 1.0) + 0.5);
                double t = N * (y_prime + 1.0);
                if (t >= 0 && t < 2 * N) {
                    int intt = static_cast<int>(t);
                    sum += attenuation[d][intt] * (intt + 1 - t) + attenuation[d][intt + 1] * (t - intt);
                }
            }
            reconstructed[i + N][j + N] = sum / Directions;
        }
    }
    outputImage(reinterpret_cast<double*>(reconstructed), "./images/reconstructed_180d_8bit_amplified_FBP_Ramp.png", true);
    outputImage(reinterpret_cast<double*>(reconstructed), "./images/reconstructed_180d_8bit_FBP_Ramp.png");
    return 0;
}