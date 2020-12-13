#sobel边缘检测
import cv2
import numpy as np

source = cv2.imread('C:/Users/123/Desktop/lena.jpg')
source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', source)
# source=source.astype(np.float32)

# sobel_x:发现垂直边缘
sobel_x = cv2.Sobel(source, cv2.CV_64F, 1, 0)
# sobel_y:发现水平边缘
sobel_y = cv2.Sobel(source, cv2.CV_64F, 0, 1)

sobel_x = np.uint8(np.absolute(sobel_x))
sobel_y = np.uint8(np.absolute(sobel_y))
np.set_printoptions(threshold=np.inf)
# print(sobel_x)

sobelCombined = cv2.bitwise_or(sobel_x, sobel_y)  # 按位或
sum = sobel_x + sobel_y

cv2.imshow('sum', sum)
cv2.waitKey()



#prewitt边缘检测
from scipy import signal


def prewitt(I, _boundary='symm', ):
    # prewitt算子是可分离的。 根据卷积运算的结合律，分两次小卷积核运算

    # 算子分为两部分，这是对第一部分操作
    # 1: 垂直方向上的均值平滑
    ones_y = np.array([[1], [1], [1]], np.float32)
    i_conv_pre_x = signal.convolve2d(I, ones_y, mode='same', boundary=_boundary)
    # 2: 水平方向上的差分
    diff_x = np.array([[1, 0, -1]], np.float32)
    i_conv_pre_x = signal.convolve2d(i_conv_pre_x, diff_x, mode='same', boundary=_boundary)

    # 算子分为两部分，这是对第二部分操作
    # 1: 水平方向上的均值平滑
    ones_x = np.array([[1, 1, 1]], np.float32)
    i_conv_pre_y = signal.convolve2d(I, ones_x, mode='same', boundary=_boundary)
    # 2: 垂直方向上的差分
    diff_y = np.array([[1], [0], [-1]], np.float32)
    i_conv_pre_y = signal.convolve2d(i_conv_pre_y, diff_y, mode='same', boundary=_boundary)

    return (i_conv_pre_x, i_conv_pre_y)


if __name__ == '__main__':
    I = cv2.imread('C:/Users/123/Desktop/lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', I)

    i_conv_pre_x, i_conv_pre_y = prewitt(I)

    # 取绝对值，分别得到水平方向和垂直方向的边缘强度
    abs_i_conv_pre_x = np.abs(i_conv_pre_x)
    abs_i_conv_pre_y = np.abs(i_conv_pre_y)

    # 水平方向和垂直方向上的边缘强度的灰度级显示
    edge_x = abs_i_conv_pre_x.copy()
    edge_y = abs_i_conv_pre_y.copy()

    # 将大于255的值截断为255
    edge_x[edge_x > 255] = 255
    edge_y[edge_y > 255] = 255

    # 数据类型转换
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    # 利用abs_i_conv_pre_x 和 abs_i_conv_pre_y 求最终的边缘强度
    # 求边缘强度有多重方法, 这里使用的是插值法
    edge = 0.5 * abs_i_conv_pre_x + 0.5 * abs_i_conv_pre_y

    # 边缘强度灰度级显示
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




#LoG进行边缘检测
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread("C:/Users/123/Desktop/lena.jpg")
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 先通过高斯滤波降噪
gaussian = cv.GaussianBlur(gray_img, (3, 3), 0)

# 再通过拉普拉斯算子做边缘检测
dst = cv.Laplacian(gaussian, cv.CV_16S, ksize=3)
LOG = cv.convertScaleAbs(dst)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图形
titles = ['原始图像', 'LOG 算子']
images = [rgb_img, LOG]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()



#canny进行边缘检测
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/123/Desktop/lena.jpg') # 读取图像

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转化为灰度图
blur = cv2.GaussianBlur(img_gray, (3, 3), 0) # 高斯滤波处理原图像降噪

canny_image = cv2.Canny(blur, 50, 150)
cv2.imshow('lena_50', img)
cv2.imshow('canny_process', canny_image)
cv2.waitKey(0)
cv2.destroyAllWindows()