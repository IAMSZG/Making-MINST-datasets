#转换工具代码
import cv2
import numpy as np
from PIL import Image
from glob import glob
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, filters, segmentation, measure, morphology, color
#print(depth_dataset.shape)
import os
i=0

# 转npy为jpg
# for filename in glob(r"C:\Users\93768\Desktop\1\data\masks\*.npy"):
#     depth_dataset = np.load(filename)
#     i+=1
#     img = depth_dataset[:,:]
#     print(depth_dataset[:,:])
    #plt.imsave("E:/data1/{:0>5d}.jpg" .format(i),  img)
    # cv2.imwrite("E:/data3/{:0>5d}.jpg".format(i), img)
# format（i）是格式{:0>5d}意思是d为i的十进制形式，>右靠齐，5为宽度也就是五个数，formit（）为放入的数，img是保存到图像
# im = Image.fromarray(depth_dataset)
# im.show()


# 转为灰度图
# path = "E:\\Data\\hsf_0"
# for dirname in os.listdir(path):
#     dirpath = path+"\\"+dirname
#     for filename in os.listdir(dirpath):
#         filepath = dirpath+"\\"+filename
#         for imgname in os.listdir(filepath):
#             imgpath = filepath + "\\" + imgname
#             for filename in glob(imgpath):
#                 i+=1
#                 img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#                 # cv2.imwrite("E:/{:0>5d}.jpg".format(i), img)
#                 cv2.imwrite(imgpath, img)


#   转为黑底，白特征的图
from PIL import Image
# import os
save_path = "E:\\Data\\32"
# path = "E:\\Data\\32image"
# for dirname in os.listdir(path):
#     dirpath = path+"\\"+dirname
#     img = cv2.imread(dirpath)
#     img = cv2.resize(img,(80,80))
#
#     element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     img = cv2.dilate(img, element)
#
#     littleImage = cv2.copyMakeBorder(img, 24, 24, 24, 24, cv2.BORDER_CONSTANT,value=0)
#     cv2.imwrite(save_path + "\\" +dirname,littleImage)


path = "E:\\Data\\32image"
for dirname in os.listdir(save_path):
    dirpath = save_path+"\\"+dirname
    # for filename in os.listdir(dirpath):
    #     filepath = dirpath+"\\"+filename
    #     for imgname in os.listdir(filepath):
    #         imgpath = filepath + "\\" + imgname
    img = Image.open(dirpath)

    # 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    Img = img.convert('L')
    # Img.save("test1.jpg")

    # 自定义灰度界限，大于这个值为白色，小于这个值为黑色
    threshold = 200

    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)

    # 图片二值化
    photo = Img.point(table, '1')
    photo.save(dirpath)

