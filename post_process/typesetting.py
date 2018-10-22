# coding:utf-8
import json
import copy

class PostProcess():
    @staticmethod
    def typesetting(text_areas):
        '''
        :param text_areas:  instance of class TextAreas
        '''
        heights = text_areas.text_areas_features()
        print heights[0], heights[1]

class TextAreas():
    def __init__(self, text_areas):
        '''
        :param text_areas:  text areas, including top left position
                            text area width height and text string
                            array of [x1, y1, x2, y2, x3, y3, x4, y4, text_string]
                            [top left, top right, bottom left, bottom right]
        '''
        self.text_areas = text_areas
        self.text_areas_x = copy.copy(text_areas).sort(key = self.get_x)
        self.text_areas_y = copy.copy(text_areas).sort(key = self.get_y)
        self.text_areas_height = copy.copy(text_areas).sort(key = self.get_height)
        self.text_areas_slope = copy.copy(text_areas).sort(key = self.get_slope)

        text_area_num = len(self.text_areas)
        self.height_average = reduce(lambda x,y: x+self.get_height(y), self.text_areas, 0)
        self.slope_average = reduce(lambda x,y: x+self.get_slope(y), self.text_areas, 0)
        self.height_average /= text_area_num
        self.slope_average /= text_area_num
        self.height_median = 0
        if text_area_num % 2 == 1:
            idx = int((text_area_num + 1) / 2) - 1
            self.height_median = self.text_areas_height[idx][3]
            self.slope_median = self.text_areas_slope[idx][3]
        else:
            idx = int(text_area_num / 2) - 1
            self.height_median = (self.text_areas_height[idx][3] + self.text_areas_height[idx + 1][3]) * 0.5
            self.height_slope = (self.text_areas_slope[idx][3] + self.text_areas_slope[idx + 1][3]) * 0.5

    def get_x(self, text_area):
        return text_area[0]
    def get_y(self, text_area):
        return text_area[1]
    def get_width(self, text_area):
        return text_area[2] - text_area[0]
    def get_height(self, text_area):
        return text_area[3] - text_area[1]
    def get_slope(self, text_area):
        return (text_area[3] - text_area[1]) / (text_area[2] - text_area[0])

    def text_area_features(self, text_area):
        '''
        calculate feature vector of one text area
        :param text_area:   text area to calculate feature vector
        :rtype:             feature vector of the text area.
                            [top left x, top left y, width, height, slope]
        '''
        return [self.get_x(text_area), self.get_y(text_area), self.get_width(text_area), self.get_height(text_area), self.get_slope(text_area)]

    def text_areas_features(self):
        '''
        calculate feature vector of entire text areas
        :rtype: feature vector of entire text areas.
                [height average, height median, slope average, slope median]
        '''
        return [self.height_average, self.height_median, self.slope_average, self.slope_median]


if __name__ == '__main__':
    json_str = ''
    with open('example.json') as f:
        lines = f.readlines()
        for line in lines:
            json_str += line
    res = json.loads(json_str)
    PostProcess.typesetting(TextAreas(res))