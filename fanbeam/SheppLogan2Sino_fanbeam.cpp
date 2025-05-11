#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;

// the Shepp and Logan phantom
// Center Coordinate (x, y), Major Axis, Minor Axis, Angle, Refractive Index
const double SHEPP_LOGAN[10][6] = 
    {{0, 0, 0.92, 0.69, 90, 2.0}, 
    {0, -0.0184, 0.874, 0.6624, 90, -0.98},
    {0.22, 0, 0.31, 0.11, 72, -0.02},
    {-0.22, 0, 0.41, 0.16, 108, -0.02},
    {0, 0.35, 0.25, 0.21, 90, 0.01},
    {0, 0.1, 0.046, 0.046, 0, 0.01},
    {0, -0.1, 0.046, 0.023, 0, 0.01},
    {-0.08, -0.605, 0.046, 0.023, 0, 0.01},
    {0, -0.605, 0.023, 0.023, 0, 0.01},
    {0.06, -0.605, 0.046, 0.023, 90, 0.01}};
const int Directions = 40;
const int Intervals = 90;
const double Pi = 3.14159265359;
const int N = 128;

double attenuation[Directions][Intervals];

int main() {
    for (int d = 0; d < Directions; d++) {
        double alpha = Pi * 2 * d / Directions;
        for (int i = 0; i < Intervals; ++i) {
            double gamma = Pi / 2 * i / Intervals - Pi / 4;
            for (int ell = 0; ell < 10; ell++) {
                double x = SHEPP_LOGAN[ell][0];
                double y = SHEPP_LOGAN[ell][1];
                double phi = alpha - gamma;
                double x_prime = x * cos(phi) + y * sin(phi);
                double y_prime = -x * sin(phi) + y * cos(phi);
                double theta = SHEPP_LOGAN[ell][4] * Pi / 180 - phi;
                double a = SHEPP_LOGAN[ell][2]; 
                double b = SHEPP_LOGAN[ell][3];
                double y1 = sqrt(2) * sin(gamma);
                double A = cos(theta) * cos(theta) / a / a + sin(theta) * sin(theta) / b / b;
                double B = 2 * cos(theta) * sin(theta) * (y1 - y_prime) * (1.0 / a / a - 1.0 / b / b);
                double C = (y1 - y_prime) * (y1 - y_prime) * (sin(theta) * sin(theta) / a / a + cos(theta) * cos(theta) / b / b) - 1;
                double delta = B * B - 4 * A * C;
                if (delta >= 0) {
                    attenuation[d][i] += sqrt(delta) / A * SHEPP_LOGAN[ell][5];
                }
            }

        }
    }
    double maxi = 0;
    for (int i = 0; i < Directions; i++) {
        for (int j = 0; j < Intervals; j++) {
            if (attenuation[i][j] > maxi) {
                maxi = attenuation[i][j];
            }
        }
    }
    cout << "Max attenuation: " << maxi << endl;
    cv::Mat sinoImage(Directions, Intervals, CV_16U);
    for (int i = 0; i < Directions; i++) {
        for (int j = 0; j < Intervals; j++) {
            sinoImage.at<uint16_t>(i, j) = static_cast<uint16_t>(attenuation[i][j] / maxi * 65535);
        }
    }
    cv::imwrite("./images/Sinogram_FanBeam_" + to_string(Directions) + "d_" + to_string(Intervals) + "i_16bit.png", sinoImage);
    return 0;
}