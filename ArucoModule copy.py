import cv2
from cv2 import aruco
import numpy as np
import os
from flask import Flask, request


def findArucoMarkers(img, markerSize=4, totalMarkers=50, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)

    arucoParam = aruco.DetectorParameters()

    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    
    # print(ids)

    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):

    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut = img + imgOut

    if drawId:
        cv2.putText(imgOut, str(id), (tl[0],tl[1]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut  


app = Flask(__name__)
imgAug = cv2.imread("corn.jpg")

@app.route('/augment', methods=['POST'])
def augment_image():
    # Convert image from binary to numpy array
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    arucoFound = findArucoMarkers(img)

    if len(arucoFound[0]) != 0:
        for bbox, id in zip(arucoFound[0], arucoFound[1]):
            img = augmentAruco(bbox, id, img, imgAug, False)

    # Convert image back to binary for response
    ret, img_encoded = cv2.imencode('.jpg', img)

    return img_encoded.tostring(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/get-image', methods=['GET'])
def getImage():
    ret, img_encoded = cv2.imencode('.jpg', imgAug)
    return img_encoded.tostring(), 200, {'Content-Type': 'image/jpeg'}


if __name__ == '__main__':
    app.run()