import cv2
import numpy as np

img0 = cv2.imread("photo_shoot_tuned/thermal/pose_0_thermal.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("photo_shoot_tuned/thermal/pose_1_thermal.png", cv2.IMREAD_GRAYSCALE)

# compute optical flow
flow = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
u, v = np.mean(flow[:,:,0]), np.mean(flow[:,:,1])
print(f"Mean optical flow from 0 to 1: u={u:.2f}, v={v:.2f}")

# if v > 0, features moved DOWN in image 1.
# Camera moved from Y=-40 to -38 (North +Y).
# Features should move South (-Y, v > 0).

