# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 4:28 PM
# @Author  : pywin
# @Email   : none
# @Project -> File : untitled -> maskMP4.py
# @Software : PyCharm

import cv2, os
import numpy as np
import re
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip


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




def get_xoy(start, imgsroot):
    path_ = os.path.join(imgsroot, 'image' + start + '.jpg')
    img = cv2.imread(path_)
    bbox = cv2.selectROI('selectroi', img)
    return bbox


def fitsign(pos, imgsroot):
    assert len(pos.split(' '))==2, print('Input Error! Use Spaces to separate them!')
    start, end = pos.split(' ')
    assert int(start)>=0 and int(end)>0, print('Input Error! input range in (0, len(total_frames)) and other symbols cannot be included')
    box = get_xoy(start, imgsroot)
    xo, yo, w, h = box[0], box[1], box[2], box[3],
    # print(box)
    dstlist = os.listdir(imgsroot)
    dstlist = sorted(dstlist, key=lambda x:int(re.findall('\d+',x)[0]), reverse = False)
    kernel_size = (41, 41)
    sigma = 50
    for n in dstlist[int(start):int(end)]:
        path_ = os.path.join(imgsroot, n)
        img = cv2.imread(path_)
        crop = img[yo:yo + h, xo:xo + w, :]
        crop = cv2.GaussianBlur(crop, kernel_size, sigma)
        # crop = cv2.blur(crop, (41,41))
        img[yo:yo + h, xo:xo + w, :] = crop
        cv2.imwrite(path_, img)


if __name__ == "__main__":
    print('='*25 + 'One stage, find the start and end positions of the image to be masked' + '='*25)
    imgsroot = './imgs_tmp'
    if not os.path.exists(imgsroot):
        os.makedirs(imgsroot)
    video_tmp = r'./a.mp4'
    video_path = r"/Users/pywin/File/datasets/deepfake/1.mov"
    video_dst = r'./{}_mask.mp4'.format(os.path.basename(video_path).split('.')[0])
    fps = video2img(video_path)
    print('=' * 25 + 'Two stage, video masked' + '=' * 25)
    pos = input('input the start and end positions of the image to be masked, separated by space:')
    fitsign(pos, imgsroot)
    img2video(video_tmp, fps)
    video_org = VideoFileClip(video_path)
    audio = video_org.audio
    video = VideoFileClip(video_tmp)  # 读入视频
    video = video.set_audio(audio)  # 将音轨合成到视频中
    video.write_videofile(video_dst)  # 输出
    os.remove(video_tmp)  # 删除临时文件
    shutil.rmtree(imgsroot)  # 删除临时文件