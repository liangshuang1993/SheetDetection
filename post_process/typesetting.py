# coding:utf-8
import json
from text_areas import TextAreas

class PostProcess():
    @staticmethod
    def typesetting(text_areas):
        '''
        :param text_areas:  instance of class TextAreas
        '''
        features = text_areas.text_areas_features()
        print features[0], features[1], features[2], features[3]

if __name__ == '__main__':
    json_str = ''
    with open('example.json') as f:
        lines = f.readlines()
        for line in lines:
            json_str += line
    res = json.loads(json_str)
    PostProcess.typesetting(TextAreas(res))