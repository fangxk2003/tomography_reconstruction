import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
import cv2

# Load matrix.csv and result.csv
print("now")
matrix = pd.read_csv('matrix.csv', header=None).values
result = pd.read_csv('result.csv', header=None).values
N, M = matrix.shape
print(N, M)
print(result.shape)
x = np.zeros(N)
print(x.shape)
phantom = shepp_logan_phantom()
phantom_resized = resize(phantom, (128, 128), mode='reflect', anti_aliasing=True)
phantom_vector = phantom_resized.flatten()
lr = 0.9
iterations = 30
directions = 180
img_n = 128
scaler = MinMaxScaler(feature_range=(0, 255))
img = np.zeros((iterations * directions, img_n, img_n), np.uint8)
img_3 = np.zeros((iterations * directions, img_n, img_n, 3), np.uint8)
for i in range(2):
    for rays in range(M):
        sum_matrix = np.sum(np.square(matrix[:, rays]))
        x += lr * (result[rays] - np.dot(matrix[:, rays], x)) * matrix[:, rays] / sum_matrix
        if ((rays + 1) % img_n == 0) : 
            img[i * directions + rays // img_n] = scaler.fit_transform(x.reshape(128, 128)).astype(np.uint8)
    mse = np.mean(np.square(x - phantom_vector))
    print("MSE:", mse)
for i in range(2):
    for d in range(directions):
        # change img[i * directions + d] to 3 channels
        img_3[i * directions + d] = cv2.merge([img[i * directions + d]] * 3)
        cv2.putText(img_3[i * directions + d],  str(i) + "," + str(d), (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.imwrite('output' + '.png', img[200])