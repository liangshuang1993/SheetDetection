# coding:utf-8
import json
import numpy as np
import cv2
import os
from text_areas import TextAreas
import draw

class PostProcess():
    col_width_threshould = 2

    @staticmethod
    def typesetting(text_areas):
        '''
        :param text_areas:  instance of class TextAreas
        '''
        features = text_areas.text_areas_features()
        print features[0], features[1], features[2], features[3]

        seperator = [0]
        sum = text_areas.get_x_of('x', 0)
        count = 1
        average = sum / count
        for i in range(1, len(text_areas.text_areas)):
            if text_areas.get_x_of('x', i) - average > features[1] * PostProcess.col_width_threshould:
                seperator.append(i)
                sum = text_areas.get_x_of('x', i)
                count = 1
            else:
                sum += text_areas.get_x_of('x', i)
                count += 1
            average = sum / count
        seperator.append(len(text_areas.text_areas))

        base_x = -1
        ft = draw.put_chinese_text('fonts/msyh.ttc')
        new_img = np.ones((3000, 3000, 3)) * 255
        color_ = (0,0,0)
        text_size = 20
        for i in range(1, len(seperator)):
            sum = 0
            count = 0
            average = 0
            for j in range(seperator[i - 1], seperator[i]):
                sum += TextAreas.get_x(text_areas.text_areas_x[j])
                count += 1
            average = sum / count
            if base_x < 0:
                base_x = average
            for j in range(seperator[i - 1], seperator[i]):
                text_areas.text_areas_x[j].append(average)

        for i in range(seperator[0], seperator[1]):
            item = text_areas.text_areas_x[i]
            item.append(TextAreas.get_y(item))

        for i in range(2, len(seperator)): # from second col
            for j in range(seperator[i - 1], seperator[i]): # for each item in second col
                item = text_areas.text_areas_x[j]
                min_dist = 10e8
                min_idx = 0
                for k in range(seperator[0], seperator[i - 1]): # from first col to i - 1 th col
                    item2 = text_areas.text_areas_x[k]
                    delta_y = abs(TextAreas.get_y(item2) - TextAreas.get_y(item))
                    if (delta_y < min_dist):
                        min_dist = delta_y
                        min_idx = k
                if (min_dist < features[1]):
                    item2 = text_areas.text_areas_x[min_idx]
                    item.append(item2[10])
                else:
                    item.append(TextAreas.get_y(item))

        
        for i in range(len(text_areas.text_areas)):
            elem = text_areas.text_areas_x[i]
            # delta = elem[0] - base_x
            # elem.append(elem[1] - delta * features[3])
            new_img = ft.draw_text(new_img, 
                                   (int(elem[9]), int(elem[10])), 
                                   elem[8], 
                                   text_size, color_)
        cv2.imwrite(os.path.join('tmp', 'image2.jpg'), new_img)
        print ''




if __name__ == '__main__':
    json_str = ''
    with open('example.json') as f:
        lines = f.readlines()
        for line in lines:
            json_str += line
    res = json.loads(json_str)
    PostProcess.typesetting(TextAreas(res))