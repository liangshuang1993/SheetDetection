import cv2
import random

image = cv2.imread("tmp1/default/00000013.jpg")

ks = [3, 5]
ksize1 = random.choice(ks)
ksize2 = random.choice(ks)
sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
sigma = 0
if ksize1 <= 3:
    sigma = random.choice(sigmas)
#img = cv2.GaussianBlur(img, (ksize1, ksize2), sigma)

print(ksize1,ksize2)
img = cv2.blur(image, (ksize1, ksize2), 1)

#image = cv2.blur(image, (30, 1), 1)
cv2.imwrite('result.png', img)
