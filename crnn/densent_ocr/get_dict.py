import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

from io import open
def readtrainfile(filename):
    res=[]
    with open(filename,'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic={}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
        print p
        #print dic
    return dic

dic = readtrainfile('/home/ls/project201803/tensorflow.image/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train1.txt')
print dic
