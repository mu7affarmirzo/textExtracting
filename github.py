import cv2
import numpy as np
import pytesseract
import os
from skimage import io
from matplotlib import pyplot as plt

per = 25

roi = [[(46, 195), (205, 214), 'text', 'Familiyasi'],
       [(47, 236), (211, 255), 'text', 'Ismi'],
       [(47, 274), (229, 293), 'text', 'Otasining ismi'],
       [(527, 236), (612, 256), 'text', 'Jinsi']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('1temp.jpg')
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)

path = 'imgs'
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + '/' + y)


    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    # imgMatch = cv2.resize(imgMatch, (w // 2, h // 2))
    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ =cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w,h))
    imgScan = cv2.resize(imgScan, (w // 3, h // 3))
    cv2.imshow(y, imgScan)



cv2.imshow('Img', imgQ)


cv2.waitKey(0)
cv2.destroyAllWindows()