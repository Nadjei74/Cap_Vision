import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np



def readAndWriteSinglePixel():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    eyeRegion = imgRGB[980:1135, 850:1150]

    dx = 1135-980
    dy = 1150 - 850

    startX = 1098
    startY = 1450

    imgRGB[startX:startX+dx, startY:startY+dy] = eyeRegion
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

def createPureColor():
    zeros = np.zeros((100,100))
    ones = np.ones((100,100))
    blmg = cv.merge((zeros, zeros, 255*ones))
    rImg = cv.merge((255*ones, zeros, zeros))
    gImg = cv.merge((zeros, 255*ones, zeros))

    plt.figure()
    plt.subplot(131)
    plt.imshow(blmg)
    plt.subplot(132)
    plt.imshow(rImg)
    plt.subplot(132)
    plt.imshow(gImg)

    plt.show()

def grayScale():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath)
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('gray', imggray)
    cv.waitKey(0)

def reSize():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    eyeRegion = img[980:1135, 850:1150, :]
    height, width,_ = eyeRegion.shape
    scale = 4

    Interps = [
        cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_NEAREST, cv.INTER_LANCZOS4
    ]

    intertitle = ['area', 'linear', 'nearest', 'cubic', 'lancoz']

    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(eyeRegion)

    for i in range(len(intertitle)):
        plt.subplot(2,3,i+2)
        imgresize = cv.resize(eyeRegion, (int(width*scale), int(height*scale)), interpolation=Interps[i])
        plt.imshow(imgresize)
        plt.title(intertitle[i])

    plt.show()

def conv2d():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    n = 100
    kernel = np.ones((n,n), np.float32)/(n*n)
    imgFilt = cv.filter2D(img, -1, kernel)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(imgFilt)

    plt.show()

def gradimage():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img,cmap='gray')



    laplace = cv.Laplacian(img, cv.CV_64F, ksize=21)
    plt.subplot(222)
    plt.imshow(laplace,cmap='gray')

    kx,ky = cv.getDerivKernels(1,0,3)
    #print(ky@kx).T
    sobelX = cv.Sobel(img, cv.CV_64F, 1,0, ksize=21)
    plt.subplot(223)
    plt.imshow(sobelX,cmap='gray')

    sobelY = cv.Sobel(img, cv.CV_64F, 0,1, ksize=21)
    plt.subplot(224)
    plt.imshow(sobelY,cmap='gray')



    plt.show()

def callback():
    pass

def cannyEdge():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width,_ = img.shape
    scale = 1/5
    height_scale = int(height*scale)
    width_scale = int(width*scale)

    img = cv.resize(img, (width_scale, height_scale), interpolation=cv.INTER_LINEAR)

    winname = 'canny'

    cv.namedWindow(winname)
    cv.createTrackbar('minthresh',winname, 0,255,callback)
    cv.createTrackbar('maxthresh',winname, 0,255,callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        minthresh = cv.getTrackbarPos('minthresh', winname)
        maxthresh = cv.getTrackbarPos('maxthresh', winname)
        cannyEdge = cv.Canny(img,minthresh,maxthresh)
        cv.imshow(winname,cannyEdge)

    cv.destroyAllWindows()


def Sift():
    imgPath =  r"C:\Users\adjei\Coding_Projects\CV\Images\cat.jpg"
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints = sift.detect(img, None)
    img = cv.drawKeypoints(img, keypoints, img, flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)   

    plt.figure()
    plt.imshow(img)
    plt.show()

        

    


if __name__ == '__main__':
    #readAndWriteSinglePixel()
    #createPureColor()
    #grayScale()
    #reSize()
    #conv2d()
    #gradimage()
    #cannyEdge()
    Sift()

