# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 4:28 PM
# @Author  : pywin
# @Email   : none
# @Project -> File : untitled -> maskMP4.py
# @Software : PyCharm

import cv2, os
import numpy as np
import re


def video2img(videoroot):
    cap = cv2.VideoCapture(videoroot)
    isOpened = cap.isOpened  # 判断是否打开‘
    print(isOpened)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)
    i = 0
    print('extract img...')
    while (isOpened):
        i += 1
        (flag, frame) = cap.read()  # 读取每一张 flag frame
        fileName = './imgs/image' + str(i) + '.jpg'

        if flag == True:
            # frame = np.rot90(frame, 1)
            cv2.imwrite(fileName, frame)
        else:
            break
    return fps


def img2video(outvideoroot, fps):
    img = cv2.imread('./imgs/image1.jpg')
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(outvideoroot, fourcc, fps, size)
    list_imgs = os.listdir('./imgs')
    for i in range(1, len(list_imgs)):
        fileName = './imgs/image' + str(i) + '.jpg'
        img = cv2.imread(fileName)
        videowriter.write(img)


def get_xoy():
    imgsroot = './imgs'
    list_ = os.listdir(imgsroot)
    for n in list_:
        path_ = os.path.join(imgsroot, n)
        img = cv2.imread(path_)
        bbox = cv2.selectROI('selectroi', img)
        break
    return bbox


def fitsign():
    box = get_xoy()
    xo, yo, w, h = box[0], box[1], box[2], box[3],
    print(box)
    dstroot = './imgs'
    dstlist = os.listdir(dstroot)
    dstlist = sorted(dstlist, key=lambda x:int(re.findall('\d+',x)[0]), reverse = False)
    kernel_size = (41, 41)
    sigma = 50
    for n in dstlist[0:25]:
        path_ = os.path.join(dstroot, n)
        img = cv2.imread(path_)
        crop = img[yo:yo + h, xo:xo + w, :]
        crop = cv2.GaussianBlur(crop, kernel_size, sigma)
        # crop = cv2.blur(crop, (41,41))
        img[yo:yo + h, xo:xo + w, :] = crop
        cv2.imwrite(path_, img)


if __name__ == "__main__":
    dstroot = './imgs'
    if not os.path.exists(dstroot):
        os.makedirs(dstroot)
    fps = video2img(r"/Users/pywin/File/datasets/deepfake/003_000.mp4")
    fitsign()
    img2video('a.avi', fps)