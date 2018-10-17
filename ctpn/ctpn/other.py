# -*- coding: utf-8 -*-  
import cv2
import numpy as np
from matplotlib import cm


def prepare_img(im, mean):
    """
        transform img into caffe's input img.
    """
    im_data = np.transpose(im - mean, (2, 0, 1))
    return im_data


def draw_boxes(im_name,
               im,
               bboxes,
               scale,
               is_display=True,
               color=None,
               caption="Image",
               wait=True):
    """
        boxes: bounding boxes
    """
    text_recs = np.zeros((len(bboxes), 8), np.int)

    im = im.copy()
    index = 0
    for box in bboxes:
        if color == None:
            if len(box) == 8 or len(box) == 9:
                c = tuple(cm.jet([box[-1]])[0, 2::-1] * 255)
            else:
                c = tuple(np.random.randint(0, 256, 3))
        else:
            c = color

        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y
        cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x1), int(y1)), (int(x3), int(y3)), c, 2)
        cv2.line(im, (int(x4), int(y4)), (int(x2), int(y2)), c, 2)
        cv2.line(im, (int(x3), int(y3)), (int(x4), int(y4)), c, 2)
        text_recs[index, 0] = int(x1/scale)
        text_recs[index, 1] = int(y1/scale)
        text_recs[index, 2] = int(x2/scale)
        text_recs[index, 3] = int(y2/scale)
        text_recs[index, 4] = int(x3/scale)
        text_recs[index, 5] = int(y3/scale)
        text_recs[index, 6] = int(x4/scale)
        text_recs[index, 7] = int(y4/scale)
        index = index + 1
        # cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c,2)
        # cv2.waitKey(0)
        # cv2.imshow('kk', im)
        cv2.imwrite(im_name,im)
        # min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        # min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        # max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        # max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

    return text_recs, im


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0] - 1)
    return boxes


def normalize(data):
    if data.shape[0] == 0:
        return data
    max_ = data.max()
    min_ = data.min()
    return (data - min_) / (max_ - min_) if max_ - min_ != 0 else data - min_


def resize_im(im, scale, max_scale=None):
    # 按照scale和图片的长宽的最小值的比值作为输入模型的图片的尺寸
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f), f
    # return cv2.resize(im, (0, 0), fx=1.2, fy=1.2), f


class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs
