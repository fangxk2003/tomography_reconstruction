import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

def sart(sinogram, theta, num_iter=10):
    n_detectors = sinogram.shape[0]
    img_size = sinogram.shape[0]
    
    # 初始化重建图像
    reconstruction = np.zeros((img_size, img_size))
    
    # 正投影和反投影算子预计算
    ones_image = np.ones((img_size, img_size))
    projector_norm = radon(ones_image, theta, circle=True)
    backprojector_norm = iradon(np.ones_like(sinogram), theta, filter_name=None, circle=True)
    
    for it in range(num_iter):
        for i, angle in enumerate(theta):
            # 单角度投影误差
            proj = radon(reconstruction, [angle], circle=True)[:, 0]
            diff = sinogram[:, i] - proj
            
            # 构造稀疏 correction sinogram，只在第 i 列有值
            correction_sino = np.zeros_like(sinogram)
            correction_sino[:, i] = diff / (projector_norm[:, i] + 1e-6)

            # 反投影更新图像
            backproj = iradon(correction_sino, theta, filter_name=None, circle=True)
            reconstruction += backproj / (backprojector_norm + 1e-6)
            
    return reconstruction

# 模拟 Shepp-Logan phantom 和其 Radon 变换
phantom = shepp_logan_phantom()
phantom = resize(phantom, (128, 128))
theta = np.linspace(0., 180., 60, endpoint=False)
sinogram = radon(phantom, theta, circle=True)

# 执行 SART 重建
reconstruction = sart(sinogram, theta, num_iter=50)

# 显示结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Phantom")
plt.imshow(phantom, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sinogram")
plt.imshow(sinogram, cmap='gray', aspect='auto')
plt.xlabel('Angle (deg)')
plt.ylabel('Detector')

plt.subplot(1, 3, 3)
plt.title("SART Reconstruction")
plt.imshow(reconstruction, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()