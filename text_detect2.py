import cv2
import numpy as np
import pytesseract
import os


path = 'imgs'
per = 25
roi = [[(45, 195), (146, 215), 'text', 'Familiyasi'],
       [(46, 234), (165, 254), 'text', 'Ismi'],
       [(46, 274), (176, 293), 'text', 'Otasining ismi']]


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('temp.jpg')
h, w, c = imgQ.shape
#imgQ = cv2.resize(imgQ, (w//2, h//2))
orb = cv2.ORB_create(10000)
kp1, des1 = orb.detectAndCompute(imgQ, None)
#imgKp1 = cv2.drawKeypoints(imgQ, kp1, None)
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    #grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #thresholdIMG = cv2.threshold(grayIMG, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #img = cv2.resize(img, (w//2, h//2))
    #cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
    #imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    #cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(img, M, (w, h))

    cv2.imshow(y, imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
    myData = []
    print(f'########################Extracting Data from Form {j}########################')

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0,255,0), 1)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        if r[2] == 'text':
            print('{} :{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
            #print(f'{r[3]} : {pytesseract.image_to_string(imgCrop)}')
            myData.append(pytesseract.image_to_string(imgCrop))

    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    cv2.imshow(y, imgShow)

    with open('DataOutput2.csv', 'a+') as f:
        for data in myData:
            f.write((str(data) + ','))
        f.write('\n')

print(myData)

#cv2.imshow("KeyPoints", imgKp1)


