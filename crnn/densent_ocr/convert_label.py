from io import open
dicts = ''
with open('/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/dicts/cn.txt', encoding='utf-8') as f:
    for ch in f.readlines():
        ch = ch.strip()
        dicts=dicts+ch

char_to_id = {j: i for i,j in enumerate(dicts)}
#print char_to_id


with open('/datasets/text_renderer/train1/default/tmp_labels.txt', encoding='utf-8') as f:
#with open('/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train2/labels.txt', encoding='utf-8') as f:
    lines = f.readlines()

#with open('/datasets/TextRecognitionDataGenerator/TextRecognitionDataGenerator/train2.txt', 'wb') as f:
with open('/datasets/text_renderer/train1/default/tmp_labels_id.txt', 'wb') as f:
    for line in lines:
        image, label = line.strip().split(' ')
        f.write(image + '.jpg')
        for char in label:
            f.write(' ' + str(char_to_id[char] + 1))
        f.write('\n')


