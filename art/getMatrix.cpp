// getdata_ART.cpp
#include <iostream>
#include <cmath>
#include <fstream>
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

const int DIRECTIONS = 40;
const int RAYS = 128;
const double PI = 3.14159265359;
const int N = 64;
const int MATRIX_N = N * N * 4;
const int MATRIX_M = RAYS * DIRECTIONS;
const double eps = 1e-6;
const double tau = 0.5 / N;

vector<vector<double>> mat(MATRIX_N, vector<double>(MATRIX_M, 0.0));
int vis[2 * N][2 * N];

int x2i(double x) {
    return int((x + 1) * N - 0.5);
}

double i2x(int i) {
    return (i + 0.5) / N - 1;
}

double area(int i, int j, double A, double B) {
    double x = i2x(i);
    double y = i2x(j);
    if (abs(A) < eps) {
        double dif = abs(y - 1.0 / B) * N;
        if (dif > 1) return 0;
        return 1 - dif;
    }
    if (abs(B) < eps) {
        double dif = abs(x - 1.0 / A) * N;
        if (dif > 1) return 0;
        return 1 - dif;
    }
    double delta = tau * sqrt(A * A + B * B);
    double c00 = ((1 - delta - B * (y - tau)) / A - x) * N;
    double c01 = ((1 + delta - B * (y - tau)) / A - x) * N;
    double c10 = ((1 - delta - A * (x - tau)) / B - y) * N;
    double c11 = ((1 + delta - A * (x - tau)) / B - y) * N;
    double c20 = ((1 - delta - B * (y + tau)) / A - x) * N;
    double c21 = ((1 + delta - B * (y + tau)) / A - x) * N;
    double c30 = ((1 - delta - A * (x + tau)) / B - y) * N;
    double c31 = ((1 + delta - A * (x + tau)) / B - y) * N;
    // cout << x << " " << y << " " << A << " " << B << endl;
    // cout << c00 << " " << c01 << " " << c10 << " " << c11 << " " << c20 << " " << c21 << " " << c30 << " " << c31 << endl;
    if (abs(c00) > 0.5) {
        if (abs(c01) < 0.5) {
            swap(c00, c01);
            swap(c10, c11);
            swap(c20, c21);
            swap(c30, c31);
        }
        else if (abs(c20) < 0.5) {
            y = -y;
            B = -B;
            swap(c00, c20);
            swap(c01, c21);
            c10 = -c10;
            c11 = -c11;
            c30 = -c30;
            c31 = -c31;
        }
        else if (abs(c21) < 0.5) {
            y = -y;
            B = -B;
            swap(c00, c20);
            swap(c01, c21);
            c10 = -c10;
            c11 = -c11;
            c30 = -c30;
            c31 = -c31;
            swap(c00, c01);
            swap(c10, c11);
            swap(c20, c21);
            swap(c30, c31);
        }
        else if (abs(c10) < 0.5) {
            swap(x, y);
            swap(A, B);
            swap(c00, c10);
            swap(c01, c11);
            swap(c20, c30);
            swap(c21, c31);
        }
        else if (abs(c11) < 0.5) {
            swap(x, y);
            swap(A, B);
            swap(c00, c10);
            swap(c01, c11);
            swap(c20, c30);
            swap(c21, c31);

            swap(c00, c01);
            swap(c10, c11);
            swap(c20, c21);
            swap(c30, c31);
        }
        else return 0;
    }

    if (abs(c10) < 0.5) {
        if (abs(c21) < 0.5 && abs(c31) < 0.5) {
            return 1 - (c00 + 0.5) * (c10 + 0.5) / 2 - (0.5 - c21) * (0.5 - c31) / 2;
        }
        if (c21 > 0.5) return 1 - (c00 + 0.5) * (c10 + 0.5) / 2;
        return (c00 + 0.5) * (c10 + 0.5) / 2;
    }
    if (abs(c30) < 0.5) {
        if (abs(c21) < 0.5 && abs(c11) < 0.5) {
            return 1 - (0.5 - c00) * (c30 + 0.5) / 2 - (c21 + 0.5) * (0.5 - c11) / 2;
        }
        if (c21 < -0.5) return 1 - (0.5 - c00) * (c30 + 0.5) / 2;
        return (0.5 - c00) * (c30 + 0.5) / 2;
    }
    // if (abs(c20) > 0.5) {
    //     cout << "error" << endl;
    //     cout << c00 << " " << c01 << " " << c10 << " " << c11 << " " << c20 << " " << c21 << " " << c30 << " " << c31 << endl;
    // }
    if ((1 - B * y) / A < x) return (c00 + 0.5) * (c20 + 0.5) / 2;
    return (0.5 - c00) * (0.5 - c20) / 2;
}

void add_mat(int i, int j, int d, int t, double ar) {
    mat[i * N * 2 + j][d * RAYS + t] = ar;
}

void calculateMatrix() {
    for (int d = 0; d < DIRECTIONS; ++d) {
        double alpha = PI * d / DIRECTIONS;
        for (int t = 0; t < RAYS; ++t) {
            double dist = (0.5 + t) / RAYS * 2 - 1;
            double A = -sin(alpha) / dist;
            double B = cos(alpha) / dist;
            queue<pair<int, int>> q;
            memset(vis, 0, sizeof(vis));
            for (int i = 0; i < N * 2; ++i) {
                if (area(i, 0, A, B) > eps) {
                    q.push({i, 0});
                    break;
                }
                if (area(i, N * 2 - 1, A, B) > eps) {
                    q.push({i, N * 2 - 1});
                    break;
                }
                if (area(0, i, A, B) > eps) {
                    q.push({0, i});
                    break;
                }
                if (area(N * 2 - 1, i, A, B) > eps) {
                    q.push({N * 2 - 1, i});
                    break;
                }
                // double x = i2x(i);
                // double y = A * x + B;
                // int j = x2i(y);
                // double yy = y;
                // int jj = j;
                // if (j < N * 2) {
                //     if (jj < 0) jj = 0;
                //     double ar = area(i, jj, A, B);
                //     while (ar > eps) {
                //         mat[i * N * 2 + jj][d * RAYS + t] = ar;
                //         ++jj;
                //         if (jj >= N * 2) break;
                //         ar = area(i, jj, A, B);
                //     }
                // }
                // j--;
                // if (j >= 0) {
                //     if (j >= N * 2) j = N * 2 - 1;
                //     double ar = area(i, j, A, B);
                //     while (ar > eps) {
                //         mat[i * N * 2 + j][d * RAYS + t] = ar;
                //         --j;
                //         if (j < 0) break;
                //         ar = area(i, j, A, B);
                //     }
                // }
            }
            if (q.empty()) {
                cout << d << " " << t << " " << A << " " << B << endl;
                cout << "error" << endl;
                exit(0);
            }
            add_mat(q.front().first, q.front().second, d, t, area(q.front().first, q.front().second, A, B));
            vis[q.front().first][q.front().second] = 1;
            while (!q.empty()) {
                pair<int, int> p = q.front();
                q.pop();
                int i = p.first;
                int j = p.second;
                for (int o = 0; o < 4; ++o) {
                    int ni = i + (o == 0 ? -1 : (o == 1 ? 1 : 0));
                    int nj = j + (o == 2 ? -1 : (o == 3 ? 1 : 0));
                    if (ni < 0 || ni >= N * 2 || nj < 0 || nj >= N * 2) continue;
                    if (vis[ni][nj]) continue;
                    vis[ni][nj] = 1;
                    double ar = area(ni, nj, A, B);
                    if (ar > eps) {
                        q.push({ni, nj});
                        add_mat(ni, nj, d, t, ar);
                    }
                }
            }
        }
    }
}

void outputMatrix() {
    ofstream outFile("matrix_40d.csv");
    for (int i = 0; i < MATRIX_N; ++i) {
        for (int j = 0; j < MATRIX_M; ++j) {
            outFile << mat[i][j];
            if (j < MATRIX_M - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();
    cout << "Matrix exported to matrix.csv" << endl;
}

int main() {
    // cout << area(N - 1, 0, .0787674, -1.00083) << endl;
    // exit(0);
    calculateMatrix();
    outputMatrix();
    return 0;
}