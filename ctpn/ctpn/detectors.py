# coding:utf-8
import sys

import numpy as np

from .cfg import Config as cfg
from .other import normalize

sys.path.append('..')
from ..lib.fast_rcnn.nms_wrapper import nms
# from lib.fast_rcnn.test import  test_ctpn

from .text_proposal_connector import TextProposalConnector


class TextDetector:
    """
        Detect text from an image
    """

    def __init__(self):
        """
        pass
        """
        self.text_proposal_connector = TextProposalConnector()

    def detect(self, img, text_proposals, scores, size, recheck_flag=False):
        """
        Detecting texts from an image
        :return: the bounding boxes of the detected texts
        """
        # text_proposals, scores=self.text_proposal_detector.detect(im, cfg.MEAN)
        keep_inds = np.where(scores > cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # nms for text proposals
        keep_inds = nms(np.hstack((text_proposals, scores)), cfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        abnormal_areas_vertex = []
        if recheck_flag:
            abnormal_areas = [] # save abnormal areas
            proposal_heights = []
            for i, box in enumerate(text_proposals):
                x1, y1, x2, y2 = box
                proposal_heights.append(abs(int(y2) - int(y1)))
            
            mediam_height = np.median(np.array(proposal_heights))
            import copy
            tmp_im = copy.deepcopy(img)
            for box in text_proposals:
                c = tuple(np.random.randint(0, 256, 3))
                x1, y1, x2, y2 = box
                # if height is bigger than 1.5*mediam_height, save that proposal to abnormal_proposals
                if abs(y2-y1) > 1.5 * mediam_height:
                    if len(abnormal_areas) == 0:
                        abnormal_areas.append([box])
                    else:
                        flag = False
                        for index in range(len(abnormal_areas)):
                            for abnormal_proposal in abnormal_areas[index]:
                                if self._proposal_distance(box, abnormal_proposal) < 120:
                                    abnormal_areas[index].append(box)
                                    flag = True
                                    break
                        if flag == False:
                            abnormal_areas.append([box])

            # get vertex of abnormal areas
            import cv2
            for area_index, area in enumerate(abnormal_areas):
                if self._recheck_proposals(area, mediam_height):
                    tmp_im2 = copy.deepcopy(img)
                    x1, y1, x2, y2 = self._get_vertex(area)
                    c = tuple(np.random.randint(0, 256, 3))
                    cv2.line(tmp_im2, (int(x1), int(y1)), (int(x2), int(y1)), c, 2)
                    cv2.line(tmp_im2, (int(x1), int(y1)), (int(x1), int(y2)), c, 2)
                    cv2.line(tmp_im2, (int(x2), int(y1)), (int(x2), int(y2)), c, 2)
                    cv2.line(tmp_im2, (int(x1), int(y2)), (int(x2), int(y2)), c, 2)
                    cv2.imwrite('area{}.jpg'.format(area_index), tmp_im2)
                    abnormal_areas_vertex.append((x1, y1, x2, y2))

        scores = normalize(scores)

        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)

        keep_inds = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_inds]

        if text_lines.shape[0] != 0:
            keep_inds = nms(text_lines, cfg.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_inds]

        return text_lines, abnormal_areas_vertex

    def filter_boxes(self, boxes):
        heights = boxes[:, 3] - boxes[:, 1] + 1
        widths = boxes[:, 2] - boxes[:, 0] + 1
        scores = boxes[:, -1]
        return np.where((widths / heights > cfg.MIN_RATIO) & (scores > cfg.LINE_MIN_SCORE) &
                        (widths > (cfg.TEXT_PROPOSALS_WIDTH * cfg.MIN_NUM_PROPOSALS)))[0]

    def _proposal_distance(self, proposal1, proposal2):
        # manhaton distance
        centor1 = [int((proposal1[0]+proposal1[2]) / 2), int((proposal1[1]+proposal1[3]) / 2)]
        centor2 = [int((proposal2[0]+proposal2[2]) / 2), int((proposal2[1]+proposal2[3]) / 2)]
        dis = abs(centor1[0] - centor2[0]) + abs(centor1[1] - centor2[1])
        return dis
    
    def _get_vertex(self, proposals):
        area_x1, area_y1, area_x2, area_y2 = proposals[0]
        for proposal in proposals:
            x1, y1, x2, y2 = proposal
            if x1 < area_x1:
                area_x1 = int(x1)
            if x2 > area_x2:
                area_x2 = int(x2)
            if y1 < area_y1:
                area_y1 = int(y1)
            if y2 > area_y2:
                area_y2 = int(y2)
        
        return [area_x1, area_y1, area_x2, area_y2]

    def _recheck_proposals(self, proposals, mediam_height):
        # check if centor of these proposals are in the same line
        # return True if these proposals are abnormal
        y = proposals[0][1]
        for proposal in proposals:
            if abs(proposal[1] - y) > mediam_height:
                return True

        return False
