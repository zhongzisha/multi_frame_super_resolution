import cv2
import numpy as np
import sys

filename = "D:\\gddata_splits\\aerial\\5120_0\\4\\4_02_02.tif"
count = int(sys.argv[1])
im = cv2.imread(filename)
H,W = im.shape[:2]
xs = np.random.randint(low=-count,high=count,size=count*2)
ys = np.random.randint(low=-count,high=count,size=count*2)
for i in range(count):
    x = xs[i]
    y = ys[i]
    yc = H//2 + y
    xc = W//2 + x
    xmin = xc - 512
    ymin = yc - 512
    xmax = xc + 512
    ymax = yc + 512
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    im1 = im[ymin:ymax, xmin:xmax, :]
    cv2.imwrite('subimg%04d.png'%i, im1)
    
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('subimg%04d_gray.png'%i, gray)