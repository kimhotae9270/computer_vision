import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np

I = np.ones([128, 128, 3]) * [0, 255, 0]  # 초록색으로 초기화

for i in range(4):
    for j in range(4):
        if j % 2 == i % 2:
            I[32 * i:32 * (i + 1), 32 * j:32 * (j + 1)] = [255, 0, 0]  # 빨간색

plt.imshow(I)
plt.show()
I_gray = cv2.cvtColor(np.uint8(I), cv2.COLOR_RGB2GRAY)

# 4. 필터 정의
filter_dx = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
filter_dy = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
filter_G = np.array([[1, 2, 1],
                     [2, 4, 2.],
                     [1, 2, 1.]])

I_x = convolve2d(I_gray, filter_dx, mode="same")
I_y = convolve2d(I_gray, filter_dy, mode="same")

Ixx = I_x ** 2
Iyy = I_y ** 2
Ixy = I_x * I_y

Sxx = convolve2d(Ixx, filter_G, mode='same')
Syy = convolve2d(Iyy, filter_G, mode='same')
Sxy = convolve2d(Ixy, filter_G, mode='same')

det = Sxx * Syy - Sxy * Sxy
trace = Sxx + Syy

R = det - 0.05 * (trace ** 2)

plt.imshow(R)
plt.show()

_, outputs, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(R > 0.01))

I_color = I.copy()
for x, y in centroids[1:]:
    cv2.circle(I_color, (int(x), int(y)), 2, (0, 0, 255), 1)  # 파란색 원으로 변경

plt.imshow(I_color / 255)
plt.title("Checkerboard with Corners")
plt.show()
