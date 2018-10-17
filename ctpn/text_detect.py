# -*- coding: utf-8 -*-  
# import tensorflow as tf
import numpy as np
from .ctpn.detectors import TextDetector
from .ctpn.model import ctpn
from .ctpn.other import draw_boxes
'''
进行文区别于识别-网络结构为cnn+rnn
'''


def text_detect(im_name, img, recheck_flag=False):
    # ctpn网络测到
    scores, boxes, img, scale = ctpn(img)
    textdetector = TextDetector()
    # draw proposals
    import copy
    tmp_im = copy.deepcopy(img)
    import cv2
    for box in boxes:
        c = tuple(np.random.randint(0, 256, 3))
        x1, y1, x2, y2 = box
        cv2.line(tmp_im, (int(x1), int(y1)), (int(x2), int(y1)), c, 2)
        cv2.line(tmp_im, (int(x1), int(y1)), (int(x1), int(y2)), c, 2)
        cv2.line(tmp_im, (int(x2), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.line(tmp_im, (int(x1), int(y2)), (int(x2), int(y2)), c, 2)
    cv2.imwrite('proposal{}.jpg'.format(int(recheck_flag)), tmp_im)

    boxes, abnormal_areas = textdetector.detect(img, boxes, scores[:, np.newaxis], img.shape[:2], recheck_flag)
    # text_recs, tmp = draw_boxes(img, boxes, caption='im_name', wait=True, is_display=False)
    text_recs, tmp = draw_boxes(
        im_name, img, boxes, scale, caption='im_name', wait=True, is_display=True)
    return text_recs, scale, img, abnormal_areas
