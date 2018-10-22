import copy

class TextAreas():
    def __init__(self, text_areas):
        '''
        :param text_areas:  text areas, including top left position
                            text area width height and text string
                            array of [x1, y1, x2, y2, x3, y3, x4, y4, text_string]
                            [top left, top right, bottom left, bottom right]
        '''
        self.text_areas = text_areas
        self.text_areas_x = copy.copy(text_areas)
        self.text_areas_x.sort(key = self.get_x)
        self.text_areas_y = copy.copy(text_areas)
        self.text_areas_y.sort(key = self.get_y)
        self.text_areas_height = copy.copy(text_areas)
        self.text_areas_height.sort(key = self.get_height)
        self.text_areas_width = copy.copy(text_areas)
        self.text_areas_width.sort(key = self.get_width)
        self.text_areas_slope = copy.copy(text_areas)
        self.text_areas_slope.sort(key = self.get_slope)

        text_area_num = len(self.text_areas)
        self.width_min = self.get_width(self.text_areas_width[0])
        self.width_max = self.get_width(self.text_areas_width[text_area_num - 1])

        text_area_num *= 1.0
        self.height_average = reduce(lambda x,y: x+self.get_height(y), self.text_areas, 0)
        self.slope_average = reduce(lambda x,y: x+self.get_slope(y), self.text_areas, 0)
        self.width_average = reduce(lambda x,y: x+self.get_width(y), self.text_areas, 0)
        self.height_average /= text_area_num
        self.slope_average /= text_area_num
        self.width_average /= text_area_num
        if text_area_num % 2 == 1:
            idx = int((text_area_num + 1) / 2) - 1
            self.height_median = self.get_height(self.text_areas_height[idx])
            self.slope_median = self.get_slope(self.text_areas_slope[idx])
            self.width_median = self.get_width(self.text_areas_width[idx])
        else:
            idx = int(text_area_num / 2) - 1
            self.height_median = (self.get_height(self.text_areas_height[idx]) + self.get_height(self.text_areas_height[idx + 1])) * 0.5
            self.slope_median = (self.get_slope(self.text_areas_slope[idx]) + self.get_slope(self.text_areas_slope[idx + 1])) * 0.5
            self.width_median = (self.get_width(self.text_areas_width[idx]) + self.get_width(self.text_areas_width[idx + 1])) * 0.5

    def get_x(self, text_area):
        return text_area[0]
    def get_y(self, text_area):
        return text_area[1]
    def get_width(self, text_area):
        return text_area[2] - text_area[0]
    def get_height(self, text_area):
        return text_area[5] - text_area[1]
    def get_slope(self, text_area):
        return (text_area[3] - text_area[1]) * 1.0 / (text_area[2] - text_area[0])

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

