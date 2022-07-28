#制作VOC格式的数据集

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from skimage import data, filters, segmentation, measure, morphology, color
# from xml.etree.ElementTree import Element, SubElement, ElementTree
# from glob import glob
# import cv2
#
# i=0
# for filename in glob(r'E:\00001.jpg'):
#     image = plt.imread(filename)
#     thresh = filters.threshold_otsu(image)  # 阈值分割
#     bw = morphology.opening(image > thresh, morphology.square(3))  # 开运算
#     cleared = bw.copy()  # 复制
#     segmentation.clear_border(cleared)  # 清除与边界相连的目标物
#     label_image = measure.label(cleared)  # 连通区域标记
#     borders = np.logical_xor(bw, cleared)  # 异或
#     label_image[borders] = -1
#     image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示s
#     annotation = Element('annotation')
#     # 生成第一个子节点 head
#     folder = SubElement(annotation, 'folder')
#     folder.text = 'VOC2007'
#     filename = SubElement(annotation, 'filename')
#     filename.text = '000001.jpg'
#     source = SubElement(annotation, 'source')
#     owner = SubElement(annotation, 'owner')
#     # 生成 root 的第二个子节点 body
#     size = SubElement(annotation, 'size')
#     width = SubElement(size, 'width')
#     width.text = '512'
#     height = SubElement(size, 'height')
#     height.text = '512'
#     depth = SubElement(size, 'depth')
#     depth.text = '3'
#     segmented = SubElement(annotation, 'segmented')
#     segmented.text = '0'
#     for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
#         # 绘制外包矩形
#         minr, minc, maxr, maxc = region.bbox
#         object = SubElement(annotation, 'object')
#         name = SubElement(object, 'name')
#         if (region.label == 1):
#             name.text = 'Hard calcification'
#         else:
#             name.text = 'Soft calcification'
#         pose = SubElement(object, 'pose')
#         pose.text = 'Unspecified'
#         truncated = SubElement(object, 'truncated')
#         truncated.text = '0'
#         difficult = SubElement(object, 'difficult')
#         difficult.text = '0'
#         bndbox = SubElement(object, 'bndbox')
#         xmin = SubElement(bndbox, 'xmin')
#         xmin.text = str(minc)
#         ymin = SubElement(bndbox, 'ymin')
#         ymin.text = str(minr)
#         xmax = SubElement(bndbox, 'xmax')
#         xmax.text = str(maxc)
#         ymax = SubElement(bndbox, 'ymax')
#         ymax.text = str(maxr)
#     i += 1
#     tree = ElementTree(annotation)
#     # 第一种
#     tree.write(r'E:/{:0>5d}.xml'.format(i), encoding='utf-8')
#     # minc maxc是横坐标的最小最大值 minr maxr是竖坐标的最小最大值 所以坐标是就是矩形的靠近原点处的角




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data, filters, segmentation, measure, morphology, color
from xml.etree.ElementTree import Element, SubElement, ElementTree
from glob import glob
import cv2
import os
i=0
dpath = "E:\\Data\\hsf_0"
path1 = "E:\\Data\\hsf_00"


def tran(dirname,filename,imgname):
    path2 = "C:\\Users\\93768\Desktop\\voc\\Annotations"
    # path2 = path1+"\\"+dirname + "\\" + filename
    # # print(path2)
    # if not os.path.exists(path2):
    #     os.makedirs(path2)
    return path2+"\\"+imgname[:-3] + "xml"


for dirname in os.listdir(dpath):
    dirpath = dpath+"\\"+dirname

    for filename1 in os.listdir(dirpath):
        filepath = dirpath+"\\"+filename1

        for imgname in os.listdir(filepath):
            imgpath = filepath + "\\" + imgname


            for filename in glob(imgpath):
                image = plt.imread(filename)
                thresh = filters.threshold_otsu(image)  # 阈值分割
                bw = morphology.opening(image > thresh, morphology.square(3))  # 开运算
                cleared = bw.copy()  # 复制
                segmentation.clear_border(cleared)  # 清除与边界相连的目标物
                label_image = measure.label(cleared)  # 连通区域标记
                borders = np.logical_xor(bw, cleared)  # 异或
                label_image[borders] = -1
                image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示s
                annotation = Element('annotation')
                # 生成第一个子节点 head
                folder = SubElement(annotation, 'folder')
                folder.text = filename1
                filename = SubElement(annotation, 'filename')
                filename.text = imgname[:-3] + "jpg"
                path = SubElement(annotation, 'path')
                path.text = imgpath
                source = SubElement(annotation, 'source')
                database = SubElement(source, 'database')
                database.text = "Unknown"
                # 生成 root 的第二个子节点 body
                size = SubElement(annotation, 'size')
                width = SubElement(size, 'width')
                width.text = '128'
                height = SubElement(size, 'height')
                height.text = '128'
                depth = SubElement(size, 'depth')
                depth.text = '1'
                segmented = SubElement(annotation, 'segmented')
                segmented.text = '0'
                for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
                    # 绘制外包矩形
                    if region.area < 150:
                        continue
                    minr, minc, maxr, maxc = region.bbox
                    minr -= 2
                    minc -= 2
                    maxr += 2
                    maxc += 2
                    object = SubElement(annotation, 'object')
                    name = SubElement(object, 'name')
                    # if (region.label == 1):
                    #     name.text = 'Hard calcification'
                    # else:
                    #     name.text = 'Soft calcification'
                    if (imgname[-5] == '0'):
                        name.text = '0'
                    if (imgname[-5] == '1'):
                        name.text = '1'
                    if (imgname[-5] == '2'):
                        name.text = '2'
                    if (imgname[-5] == '3'):
                        name.text = '3'
                    if (imgname[-5] == '4'):
                        name.text = '4'
                    if (imgname[-5] == '5'):
                        name.text = '5'
                    if (imgname[-5] == '6'):
                        name.text = '6'
                    if (imgname[-5] == '7'):
                        name.text = '7'
                    if (imgname[-5] == '8'):
                        name.text = '8'
                    if (imgname[-5] == '9'):
                        name.text = '9'

                    pose = SubElement(object, 'pose')
                    pose.text = 'Unspecified'
                    truncated = SubElement(object, 'truncated')
                    truncated.text = '0'
                    difficult = SubElement(object, 'difficult')
                    difficult.text = '0'
                    bndbox = SubElement(object, 'bndbox')
                    xmin = SubElement(bndbox, 'xmin')
                    xmin.text = str(minc)
                    ymin = SubElement(bndbox, 'ymin')
                    ymin.text = str(minr)
                    xmax = SubElement(bndbox, 'xmax')
                    xmax.text = str(maxc)
                    ymax = SubElement(bndbox, 'ymax')
                    ymax.text = str(maxr)
                i += 1
                tree = ElementTree(annotation)
                # 第一种
                tree.write(tran(dirname,filename1,imgname), encoding='utf-8')
                # minc maxc是横坐标的最小最大值 minr maxr是竖坐标的最小最大值 所以坐标是就是矩形的靠近原点处的角
