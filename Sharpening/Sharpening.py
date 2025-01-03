import cv2
import numpy as np

ori_img = cv2.imread('input.jpg') / 255

h, w, _ = ori_img.shape
rate = h / 512
img = cv2.resize(ori_img,(int(w / rate), 512))

cv2.imshow('input', img)
box_filter = np.ones((3, 3)) / 9.0

h, w, _ = img.shape
out = np.zeros([h - 2, w - 2, 3])

for i in range(h - 2):
    for j in range(w - 2):
        for k in range(3):
            blurred_pixel = np.sum(box_filter * img[i:i + 3, j:j + 3, k])
            sharpened_pixel = img[i + 1, j + 1, k] + (img[i + 1, j + 1, k] - blurred_pixel)
            out[i, j, k] = sharpened_pixel

cv2.imshow('output', out)
cv2.waitKey(0)
