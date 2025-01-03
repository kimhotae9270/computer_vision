import cv2
import numpy as np

img1 = cv2.imread('Eiffel.jpg')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('EiffelT2.png')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)



sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(gray1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2,None)


src = []
dst = []

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches,key = lambda x:x.distance)
for match in matches:
    k1 = match.queryIdx
    k2 = match.trainIdx
    x1,y1 = keypoints_1[k1].pt
    x2,y2 = keypoints_2[k2].pt
    src.append([x1, y1])
    dst.append([x2, y2])

img3 = cv2.drawMatches(img1,keypoints_1,img2,keypoints_2,matches[:10],None)
img1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
img2 = cv2.drawKeypoints(gray2,keypoints_2,img2)

M,mask = cv2.findHomography(np.float32(src),np.float32(dst),cv2.RANSAC,5.0)
(H,W) = gray1.shape
src_corner = np.array([[0,0],[0,H-1],[W-1,H-1],[W-1,0]]).reshape(-1,1,2).astype(np.float32)
dst_corner = cv2.perspectiveTransform(src_corner,M)



img_matches = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Keypoint Matches", img_matches)
cv2.imshow("Transformed Image with Detected Corners", cv2.polylines(img2,[np.int32(dst_corner)],True,(255,255,255),2,cv2.LINE_AA))
cv2.waitKey(0)
cv2.destroyAllWindows()