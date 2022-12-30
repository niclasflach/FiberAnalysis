import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN = 200
image = "60min.jpg"
img = cv2.imread(image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (255,45,45)
text_thickness = 2
kernel = np.ones((2, 2), np.uint8)

# dilate and threshold
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(gray, kernel, iterations=1)
ret, thresh = cv2.threshold(dilated, 245, 255, cv2.THRESH_BINARY)
thresh = cv2.dilate( thresh, kernel, iterations=6)
#ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#thresh = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))

thresh_type = (thresh/255).astype(np.uint8)


#dist = cv2.distanceTransform(thresh_type, cv2.DIST_L2, 3)
#dist = cv2.distanceTransform(thresh_type, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#masken = np.zeros(img.shape, dtype="uint8")
img_temp = cv2.imread(image)
for contour in contours:
    #img_temp = cv2.imread('xlinkimage.jpg')
    if len(contour) > MIN:
        #masken = np.zeros(img.shape, np.uint8)
        masken = np.zeros_like(thresh)
        cv2.drawContours(masken, [contour], -1 , (255),-1)
        cv2.drawContours(img_temp, [contour], -1 , (255),2)
        result = cv2.distanceTransform(masken, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        #print(max_val)
        #print(max_loc)
        result2 = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        text_pos = (max_loc[0] + 50, max_loc[1] - 50)
        img_temp = cv2.putText(img_temp, str(round(max_val*2*2.78, 2)),text_pos, font, 1 , text_color,text_thickness, cv2.LINE_AA )
        cv2.circle(img_temp, max_loc , 3,(0,255,0),1)
        cv2.line(img_temp,text_pos,max_loc,text_color,2)

        cv2.imshow('Distance transform', img_temp)
        #current_contour = cv2.bitwise_and(dist,dist, mask=masken)
        #cv2.imshow('Contour', current_contour)
        #cv2.waitKey(0)

#cv2.imshow('Masked out', current_contour)
#cv2.imshow('Mask',mask)
#cv2.imshow(f'fiber', img)
#cv2.imshow('dialated', dilated)
#cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
