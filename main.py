import cv2
import numpy as np
import pytesseract
import os


per = 25

roi = [[(67, 294), (217, 324), 'text', 'Familiyasi'],
       [(217, 323), (68, 354), 'text', 'Ismi'],
       [(245, 384), (70, 415), 'text', 'Otasining ismi'],
       [(264, 445), (70, 470), 'text', "Tug'ilgan sanasi"],
       [(265, 500), (793, 356), 'text', 'Jinsi'],
       [(397, 473), (563, 504), 'text', "Tug'ilgan joyi"],
       [(69, 573), (541, 635), 'text', 'Kim tomonidan berilgan'],
       [(741, 796), (967, 828), 'text', 'Passport raqami'],
       [(69, 524), (268, 553), 'text', 'Millati']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('temp.jpg')
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