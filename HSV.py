import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, filters, segmentation, measure, morphology, color
import os
import numpy as np

if __name__ == "__main__":
    # 替换照片，测试
    # img = cv2.imread('D:\\jl.jpg')
    img = cv2.imread('D:\\yuantu\\1.jpg')
    #
    cv2.imshow("original image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 裁剪
    # wMax = img.shape[1]
    # hMax = img.shape[0]
    # xmin, ymin, w, h = 0, 0, wMax//5, hMax  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
    # img = img[ymin:ymin + h, xmin:xmin + w].copy()  # 切片获得裁剪后保留的图像区域
    #
    # cv2.imshow("DemoCrop", img)  # 在窗口显示 彩色随机图像
    # key = cv2.waitKey(0)  # 等待按键命令


    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # H、S、V范围一：
    lower1 = np.array([0, 43, 46])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)
    # cv2.imshow("mask1", mask1)  # 单通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # H、S、V范围二：
    lower2 = np.array([156, 43, 46])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)
    # cv2.imshow("mask2", mask2)  # 单通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2
    # 结果显示
    cv2.imshow("mask3", mask3)    #单通道
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 膨胀处理（可多次膨胀）
    image = cv2.dilate(mask3, element)
    image = cv2.dilate(image, element)
    cv2.imshow("pz", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresh = filters.threshold_otsu(image)  # 阈值分割
    bw = morphology.opening(image > thresh, morphology.square(3))  # 开运算
    cleared = bw.copy()  # 复制
    segmentation.clear_border(cleared)  # 清除与边界相连的目标物
    label_image = measure.label(cleared)  # 连通区域标记
    borders = np.logical_xor(bw, cleared)  # 异或
    label_image[borders] = -1
    image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
    ax0.imshow(cleared, plt.cm.gray)
    ax1.imshow(image_label_overlay)
    i = 0
    Data_set = []
    for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集

         # 绘制外包矩形
         minr, minc, maxr, maxc = region.bbox
         rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   # minc maxc是横坐标的最小最大值 minr maxr是竖坐标的最小最大值 所以坐标是就是矩形的靠近原点处的角
                                   fill=False, edgecolor='red', linewidth=1)

         # 存储质心
         Data_set.append(region.centroid)
         # 过滤小区域
         if region.area < 100:
             continue
         # 截成小图
         xmin, ymin, w, h = minc, minr, maxc - minc, maxr - minr  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
         littleImage = image[ymin:ymin + h, xmin:xmin + w].copy()  # 切片获得裁剪后保留的图像区域
         # cv2.imshow("LittleImg", littleImage)  # 在窗口显示 彩色随机图像
         # key = cv2.waitKey(0)  # 等待按键命令
         littleImage = cv2.copyMakeBorder(littleImage, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)  #给周围扩大一圈6个0值像素,上下左右

         #变成边相等的图片
         cha = max(w,h) - min(w,h)
         if cha % 2 == 1:
             add = (cha - 1) // 2
             youadd = add + 1
             if min(w,h) == w:
                 finalImage = cv2.copyMakeBorder(littleImage, 0, 0, add, youadd, cv2.BORDER_CONSTANT, value=0)
             else:
                 finalImage = cv2.copyMakeBorder(littleImage, youadd, add, 0, 0, cv2.BORDER_CONSTANT, value=0)
         else:
             add = cha // 2
             if min(w, h) == w:
                 finalImage = cv2.copyMakeBorder(littleImage, 0, 0, add, add, cv2.BORDER_CONSTANT, value=0)
             else:
                 finalImage = cv2.copyMakeBorder(littleImage, add, add, 0, 0, cv2.BORDER_CONSTANT, value=0)

         #resize到28*28的图片
         dst = cv2.resize(finalImage, (28, 28))
         #保存
         i += 1
         cv2.imwrite(r'D:\\test\\{:0>5d}.jpg'.format(i),dst)
         print(minr, minc, maxr, maxc)
         ax1.add_patch(rect)
    print(Data_set)
    fig.tight_layout()
    plt.show()


