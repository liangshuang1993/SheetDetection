import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

from io import open

key_file = '/media/NewSSD/workspace/text_renderer-dev/data/chars/med.txt'
new_key_file = '/media/NewSSD/workspace/text_renderer-dev/data/chars/med2.txt'
corpus_file = '/media/NewSSD/workspace/text_renderer-dev/corpus.txt'

def readfile(filename):
    res=[]
    with open(filename,'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    return res

def generate_new_key():
    old_key = readfile(key_file)
    corpus = readfile(corpus_file)
    dic={}
    for label in corpus:
        for char in label:
            if char not in old_key:
                old_key.append(char)
                # print(char)
    with open(new_key_file, 'w') as f:
        for key in old_key:
            f.write(key + '\n')


generate_new_key()