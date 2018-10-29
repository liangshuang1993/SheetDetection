# coding:utf-8
##添加文本方向 检测模型，自动检测文字方向，0、90、180、270
from math import *

import cv2
import os
import shutil
import numpy as np
from PIL import Image
import sys
import copy
import json


sys.path.append("crnn")
# from angle.predict import predict as angle_detect  ##文字方向检测


from ctpn.text_detect import text_detect
from crnn.model import predict as crnn_predict

import draw

RESULTS = 'results'
RESULTS_CTPN = os.path.join(RESULTS, 'ctpn')
if os.path.exists(RESULTS):
    shutil.rmtree(RESULTS)
os.makedirs(RESULTS)
os.makedirs(RESULTS_CTPN)

IMAGE_SIZE = np.array((400, 600, 3), dtype=np.int32)

def ruihua(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # retval2,thre2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  
    # return thre2
    imgAdapt=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    return imgAdapt

def crnnRec(imname, im, text_recs, ocrMode='keras', adjust=False):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    index = 0
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]
    
    if not os.path.exists(os.path.join(RESULTS_CTPN, imname.split('.')[0])):
        os.makedirs(os.path.join(RESULTS_CTPN, imname.split('.')[0]))

    for index, rec in enumerate(text_recs):
        results[index] = [
            rec,
        ]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2),
                   min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  ##图像倾斜角度

        partImg = dumpRotateImage(im, degree, pt1, pt2, pt3, pt4)

        min_x = max(min(rec[0], rec[2], rec[4], rec[6]), 0)
        min_y = max(min(rec[1], rec[3], rec[5], rec[7]), 0)
        max_x = max(rec[0], rec[2], rec[4], rec[6], 0)
        max_y = max(rec[1], rec[3], rec[5], rec[7], 0)
        partImg = im[min_y: max_y, min_x: max_x, :]
        try:
            # 根据ctpn进行识别出的文字区域，进行不同文字区域的crnn识别
            image = Image.fromarray(partImg).convert('L')
            # image = Image.fromarray(ruihua(partImg))
            image.save(os.path.join(RESULTS_CTPN, imname.split('.')[0], str(index) + '.png'))
            # cv2.imwrite(os.path.join(RESULTS_CTPN, imname.split('.')[0], str(index) + '.png'), image)
            # 进行识别出的文字识别
            sim_pred = crnn_predict(image)
            results[index].append(sim_pred)  ##识别文字
        except Exception:
            results[index].append('Error')

    return results


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
                         max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]
    return imgOut


def model(index, im_name, img, model='keras', adjust=True, recheck_flag=True):
    """
    @@param:img,
    @@param:model,选择的ocr模型，支持keras\pytorch版本
    @@param:adjust 调整文字识别结果
    
    """
    original_img = img
    # 进行图像中的文字区域的识别
    text_recs, scale, img, abnormal_areas=text_detect(os.path.join(RESULTS_CTPN, im_name.split('.')[0] + str(index) + '.png'), 
                                                    img, recheck_flag=recheck_flag)
    if len(abnormal_areas):
        print abnormal_areas[0]
        #有异常区域，需要重新检测,暂时只检测第一个
        #将异常区域贴到新的图片上，防止异常区域过小
        abnormal_img = Image.fromarray(img[int(abnormal_areas[0][1]): int(abnormal_areas[0][3]), int(abnormal_areas[0][0]): int(abnormal_areas[0][2]), :])
        sub_size = max(abnormal_img.size[0], abnormal_img.size[1])
        if sub_size < 200:
            sub_img = Image.new('RGB', (200, 200), (255, 255, 255))
        else:
            sub_img = Image.new('RGB', (sub_size, sub_size), (255, 255, 255))
        sub_img.paste(abnormal_img, (0, 0))
        sub_img.save('subimage.jpg')
        sub_text_recs, sub_scale, sub_img, _=text_detect(os.path.join(RESULTS_CTPN, im_name.split('.')[0] + str(index) + 'sub.png'),
                                                       np.array(sub_img.convert('RGB')), recheck_flag=False)

        finally_text_recs = []
        # 把大图片和小图片的text_rec合并
        # 大图片
        for rec in text_recs:
            for i in range(4):
                if not rec_in_area(rec, abnormal_areas[0], scale):
                    finally_text_recs.append(rec)
        # 小图片
        for rec in sub_text_recs:
            #绝对位置
            abs_rec = []
            for i in range(4):
                abs_rec.append(int(rec[2 * i] / scale + abnormal_areas[0][0] / scale))
                abs_rec.append(int(rec[2 * i + 1] / scale + abnormal_areas[0][1] / scale))

            finally_text_recs.append(abs_rec)
    else:
        finally_text_recs = text_recs
    
    draw_box(im_name, original_img, finally_text_recs)
    
    # 识别区域排列
    # text_recs = sort_box(text_recs)
    # 
     
    # results = crnnRec(im_name, original_img, text_recs, model, adjust=adjust)
    results = crnnRec(im_name, original_img, finally_text_recs, model, adjust=adjust)

    # draw new image
    new_img = np.ones(original_img.shape) * 255
    # new_img = np.ones(IMAGE_SIZE) * 255
    color_ = (0,0,0)
    text_size = 20
    ft = draw.put_chinese_text('/datasets/text_renderer/data/fonts/chn/songti.ttf')
    f_result = open('results/{}.txt'.format(im_name), 'w')
    f_json = open('results/{}.json'.format(im_name.split('.')[0]), 'w')
    result_dict = []
    for result in results.items():
        box, b = result[1]
        pos = (box[0], box[1])
        print b[0]
        #print box
        new_img = ft.draw_text(new_img, pos, b[0], text_size, color_)
        #f_result.write('{}  {}, {} \n'.format(b[0], pos[0], pos[1]))
        result_dict.append([box[0], box[1], box[2]-box[0], box[5] - box[3], b[0]])
    cv2.imwrite(os.path.join(RESULTS, im_name), new_img)      
    f_json.write(json.dumps(result_dict, ensure_ascii=False))    
 
    return


def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def rec_in_area(rec, abnormal_area, scale):
    abnormal_area = [x / scale for x in abnormal_area]
    x1, y1, x2, y2 = abnormal_area
    for i in range(4):
        x = rec[2 * i]
        y = rec[2 * i + 1]
        if x > x1 and x < x2 and y > y1 and y < y2:
            return True
    x1 = min(rec[0], rec[4])
    y1 = min(rec[1], rec[3])
    x2 = max(rec[2], rec[6])
    y2 = max(rec[5], rec[7])
    
    if abnormal_area[0] > x1 and abnormal_area[0] < x2 and abnormal_area[1] > y1 and abnormal_area[1] < y2:
        return True
    return False

def draw_box(im_name, original_img, text_recs):
    im = copy.deepcopy(original_img)
    c = tuple(np.random.randint(0, 256, 3))
    for rec in text_recs:
        x1, y1, x2, y2, x3, y3, x4, y4 = rec
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)), c, 2)
        cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), c, 2)
        cv2.imwrite(im_name,im)
        
