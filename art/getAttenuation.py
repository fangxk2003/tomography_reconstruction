import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import imageio
import pandas as pd
import matplotlib.pyplot as plt

phantom = shepp_logan_phantom()
phantom_resized = resize(phantom, (128, 128), mode='reflect', anti_aliasing=True)
phantom_vector = phantom_resized.flatten()
matrix = pd.read_csv('matrix_40d.csv', header=None).values
# print("Matrix shape:", matrix.shape)
# while True :
#     d, t = input().split()
#     d = int(d)
#     t = int(t)
#     plt.imshow(matrix[:, d * 128 + t].reshape(128, 128), cmap='gray')
#     plt.axis('off')
#     plt.show()
# Perform matrix multiplication
result = np.dot(phantom_vector, matrix)
print("Result shape:", result.shape)
# save the result as csv
np.savetxt('result_40d.csv', result, delimiter=',')