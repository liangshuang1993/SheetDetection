# coding:utf-8
import json

def choose_x(elem):
    return elem[0]
def choose_y(elem):
    return elem[1]
def choose_w(elem):
    return elem[3]
def choose_h(elem):
    return elem[4]

def typesetting(text_areas):
    '''
    :param text_areas:  text areas, including top left position
                        text area width height and text string
                        array of [x, y, w, h, text_string]
    '''
    text_area_num = len(text_areas)
    text_areas.sort(key = choose_h)
    height_median = 0
    height_average = reduce(lambda x,y: x+y[3], text_areas, 0)
    height_average /= text_area_num
    if text_area_num % 2 == 1:
        idx = int((text_area_num + 1) / 2) - 1
        height_median = text_areas[idx][3]
    else:
        idx = int(text_area_num / 2) - 1
        height_median = (text_areas[idx][3] + text_areas[idx + 1][3]) * 0.5
    text_areas.sort(key = choose_y)
    print height_median, height_average
    print json.dumps(text_areas)

if __name__ == '__main__':
    json_str = ''
    with open('example.json') as f:
        lines = f.readlines()
        for line in lines:
            json_str += line
    res = json.loads(json_str)
    typesetting(res)