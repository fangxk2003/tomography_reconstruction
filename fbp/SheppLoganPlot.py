import matplotlib.pyplot as plt
import numpy as np

# the Shepp and Logan phantom
# Center Coordinate (x, y), Major Axis, Minor Axis, Angle, Refractive Index
ellipses = np.array([[0, 0, 0.92, 0.69, 90, 2.0], 
                     [0, -0.0184, 0.874, 0.6624, 90, -0.98],
                     [0.22, 0, 0.31, 0.11, 72, -0.02],
                     [-0.22, 0, 0.41, 0.16, 108, -0.02],
                     [0, 0.35, 0.25, 0.21, 90, 0.01],
                     [0, 0.1, 0.046, 0.046, 0, 0.01],
                     [0, -0.1, 0.046, 0.023, 0, 0.01],
                     [-0.08, -0.605, 0.046, 0.023, 0, 0.01],
                     [0, -0.605, 0.023, 0.023, 0, 0.01],
                     [0.06, -0.605, 0.046, 0.023, 90, 0.01]], dtype=np.float32)

# BRUTEFORCE
# ppi = 100  # pixels per inch
# canvas = np.zeros((ppi * 2, ppi * 2), dtype=np.float32)
# for ellipse in ellipses:
#     print('dd')
#     x, y, a, b, theta, refractive_index = ellipse
#     theta = np.deg2rad(theta)
#     for i in range(ppi * 2):
#         for j in range(ppi * 2):
#             y_prime = -(j / ppi - 1 - x) * np.sin(theta) + (1 - (i + 1) / ppi - y) * np.cos(theta)
#             x_prime = (j / ppi - 1 - x) * np.cos(theta) + (1 - (i + 1)/ ppi - y) * np.sin(theta)
#             if (x_prime / a) ** 2 + (y_prime / b) ** 2 <= 1:
#                 canvas[i, j] += refractive_index

# USING DIFFERENCE ACCUMULATION TO ACCELERATE
ppi = 1000  # pixels per inch
canvas = np.zeros((ppi * 2, ppi * 2), dtype=np.float32)
alpha = 0
for ellipse in ellipses:
    print('dd')
    x0, y0, a, b, theta, refractive_index = ellipse
    x_ = x0 * np.cos(alpha) + y0 * np.sin(alpha)
    y_ = -x0 * np.sin(alpha) + y0 * np.cos(alpha)
    x0 = x_
    y0 = y_
    theta = np.deg2rad(theta)
    theta -= alpha
    for j in range(ppi * 2):
        x = j / ppi - 1 - x0
        A = (np.sin(theta) / a) ** 2 + (np.cos(theta) / b) ** 2
        B = 2 * x * np.cos(theta) * np.sin(theta) * (1 / a ** 2 - 1 / b ** 2)
        C = (x * np.cos(theta) / a) ** 2 + (x * np.sin(theta) / b) ** 2 - 1
        delta = B ** 2 - 4 * A * C
        if delta < 0:
            continue
        y1 = (-B + np.sqrt(delta)) / (2 * A)
        y2 = (-B - np.sqrt(delta)) / (2 * A)
        i1 = int((1 - y1 - y0) * ppi) - 1
        i2 = int((1 - y2 - y0) * ppi) - 1
        assert(i1 >= 0 and i1 < ppi * 2 and i2 >= 0 and i2 < ppi * 2 - 1)
        canvas[i1, j] += refractive_index
        canvas[i2 + 1, j] -= refractive_index
for j in range(ppi * 2):
    for i in range(1, ppi * 2):
        canvas[i, j] += canvas[i - 1, j]
    
# img = np.zeros((ppi * 2, ppi * 2), dtype=np.uint8)
# for i in range(ppi * 2):
#     for j in range(ppi * 2):
#         if canvas[i, j] < 1: img[i, j] = 0
#         elif canvas[i, j] > 1.05: img[i, j] = 255
#         else: img[i, j] = int((canvas[i, j] - 1) / 0.05 * 255)


# canvas = ((canvas - canvas.min()) / (canvas.max() - canvas.min()) * 65535).astype(np.uint16)
# # Histogram equalization
# hist, bins = np.histogram(canvas.flatten(), bins=65536, range=[0, 65535])
# cdf = hist.cumsum()
# cdf_normalized = cdf * (65535 / cdf[-1])
# canvas_equalized = np.interp(canvas.flatten(), bins[:-1], cdf_normalized).reshape(canvas.shape)
# print(canvas_equalized.dtype)

plt.imshow(canvas / canvas.max() * 255, cmap='gray', extent=[-1, 1, -1, 1])
plt.gca().set_aspect('equal')  # set the aspect ratio of the plot to be equal
plt.title("Ellipse using matplotlib")
plt.grid(True)
plt.show()
