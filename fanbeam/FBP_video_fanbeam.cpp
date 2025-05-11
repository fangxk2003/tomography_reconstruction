#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>
#include <fftw3.h>

using namespace std;
const int Directions = 360;
const int Intervals = 256;
const double Pi = 3.14159265359;
const int N = 128;
const int GrayLevel = 5000;
double attenuation[Directions][Intervals];
double reconstructed[Directions][2 * N][2 * N];
uint8_t outputVideo[Directions][2 * N][2 * N];

void inputImage() {
    cv::Mat inputImage = cv::imread("./images/Sinogram_FanBeam_" + to_string(Directions) + "d_" + to_string(Intervals) + "i_16bit.png", cv::IMREAD_UNCHANGED);
    if (inputImage.empty()) {
        cerr << "Failed to load input image!" << endl;
        exit(1);
    }

    if (inputImage.rows != Directions || inputImage.cols != Intervals) {
        cerr << "Input image dimensions do not match expected size!" << endl;
        exit(1);
    }

    for (int i = 0; i < Directions; ++i) {
        for (int j = 0; j < Intervals; ++j) {
            attenuation[i][j] = static_cast<double>(inputImage.at<uint16_t>(i, j));
        }
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
    for (int d = 0; d < Directions; ++d) {
        for (int i = 0; i < Intervals; ++i) {
            double gamma = Pi / 2 * i / Intervals - Pi / 4;
            if (i != Intervals / 2) {
                attenuation[d][i] *= cos(gamma) * (gamma * gamma / sin(gamma) / sin(gamma));
            }
        }
    }
    for (int i = 0; i < Directions; ++i) {
        applyFilter(attenuation[i], Intervals);
    }
    for (int i = -N; i < N; ++i) {
        for (int j = -N; j < N; ++j) {
            double x = static_cast<double>(i) / N;
            double y = static_cast<double>(j) / N;
            double sum = 0.0;
            for (int d = 0; d < Directions; ++d) {
                double alpha = Pi * 2 * d / Directions + Pi / 2;
                double x_prime = x * cos(alpha) + y * sin(alpha);
                double y_prime = -x * sin(alpha) + y * cos(alpha);
                double gamma = atan(y_prime / (sqrt(2) - x_prime)) / Pi * 2 * Intervals + Intervals / 2;
                if (gamma >= 0 && gamma < Intervals - 1) {
                    int intgamma = static_cast<int>(gamma);
                    reconstructed[d][i + N][j + N] = attenuation[d][intgamma] * (intgamma + 1 - gamma) + attenuation[d][intgamma + 1] * (gamma - intgamma);
                }
            }
        }
    }
    for (int d = 1; d < Directions; ++d) {
        for (int i = -N; i < N; ++i) {
            for (int j = -N; j < N; ++j) {
                reconstructed[d][i + N][j + N] += reconstructed[d - 1][i + N][j + N];
            }
        }
    }
    double maximum = 0.0;
    for (int i = 0; i < 2 * N; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            if (reconstructed[Directions - 1][i][j] > maximum) {
                maximum = reconstructed[Directions - 1][i][j];
            }
        }
    }
    vector<int> histogram(GrayLevel, 0);
    for (int i = 0; i < 2 * N; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            int value = static_cast<int>(reconstructed[Directions - 1][i][j] / maximum * GrayLevel);
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
    for (int d = 0; d < Directions; ++d) {
        for (int i = 0; i < 2 * N; ++i) {
            for (int j = 0; j < 2 * N; ++j) {
                int value = static_cast<int>((reconstructed[d][i][j] / (d + 1) * Directions / maximum * GrayLevel - L) / (R - L) * 255.0);
                value = std::max(0, std::min(255, value)); // Clamp to [0, 255]
                outputVideo[d][i][j] = static_cast<uint8_t>(value);
            }
        }
    }
    cv::VideoWriter videoWriter("./images/reconstructed_360d_256i_8bit_fanbeam_video.avi", 
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                30, 
                                cv::Size(2 * N, 2 * N), 
                                false);

    if (!videoWriter.isOpened()) {
        cerr << "Failed to open video file for writing!" << endl;
        return 1;
    }

    for (int d = 0; d < Directions; ++d) {
        cv::Mat frame(2 * N, 2 * N, CV_8UC1, outputVideo[d]);
        videoWriter.write(frame);
    }

    videoWriter.release();
    return 0;
}