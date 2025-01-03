import cv2
import numpy as np

ori_img = cv2.imread('input.jpg') / 255
h, w, _ = ori_img.shape

rate = h / 512
img = cv2.resize(ori_img, (int(w / rate), 512))

cv2.imshow('input', img)

sobel_filter_x = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
sobel_filter_y = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

h, w, _ = img.shape
out_x = np.zeros([h - 2, w - 2, 3])
out_y = np.zeros([h - 2, w - 2, 3])

for i in range(h - 2):
    for j in range(w - 2):
        for k in range(3):
            gx = np.sum(sobel_filter_x * img[i:i + 3, j:j + 3, k])
            out_x[i, j, k] = gx
            gy = np.sum(sobel_filter_y * img[i:i + 3, j:j + 3, k])
            out_y[i,j,k] = gy

cv2.imshow('output_x', out_x)
cv2.imshow('output_y', out_y)
cv2.waitKey(0)
