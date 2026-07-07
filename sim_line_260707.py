import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

img = mpimg.imread('solidWhiteCurve.jpg')

plt.figure(figsize=(10, 8))
print('This image is : ', type(img), 'with dimensions : ', img.shape)
plt.imshow(img)
plt.show()

# 1) Grayscale — mpimg는 RGB로 읽으므로 RGB2GRAY 사용
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = grayscale(img)
plt.figure(figsize=(10, 8))
plt.imshow(gray, cmap='gray')          # 수정 1
plt.show()

# 2) Gaussian Blur
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)   # 수정 2

kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

plt.figure(figsize=(10, 8))            # 공백 제거
plt.imshow(blur_gray, cmap='gray')
plt.show()

# 3) Canny Edge
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

low_threshold = 50
high_threshold = 150                   # 1:3 비율 권장 (아래 설명 참고)
edges = canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10, 8))
plt.imshow(edges, cmap='gray')
plt.show()

# 4) ROI 마스크
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

imshape = img.shape
print(imshape)
vertices = np.array([[(100, imshape[0]), (450, 320),
                      (550, 320), (imshape[1] - 20, imshape[0])]], dtype=np.int32)
masked = region_of_interest(edges, vertices)

plt.figure(figsize=(10, 8))
plt.imshow(masked, cmap='gray')
plt.show()

# 5) Hough 변환으로 직선 검출
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    if lines is None:                  # None 방어
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,   # 수정 4
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # 수정 5
    draw_lines(line_img, lines)
    return line_img

rho = 2
theta = np.pi / 180
threshold = 90
min_line_len = 120
max_line_gap = 150

lines = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)

plt.figure(figsize=(10, 8))
plt.imshow(lines)
plt.show()

# 6) 원본 + 검출선 합성
def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv2.addWeighted(initial_img, a, img, b, c)

lines_edges = weighted_img(lines, img, a=0.8, b=1., c=0.)
plt.figure(figsize=(10, 8))
plt.imshow(lines_edges)
plt.show()