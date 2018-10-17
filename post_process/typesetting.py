import json


def typesetting(text_areas):
    '''
    :param text_areas:  text areas, including top left position
                        text area width height and text string
                        array of [x, y, w, h, text_string]
    '''

if __name__ == '__main__':
    json_str = ''
    with open('result.json') as f:
        lines = f.readlines()
        for line in lines:
            json_str += line
    res = json.loads(json_str)
    typesetting(res)