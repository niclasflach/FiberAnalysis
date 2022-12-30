#https://stackoverflow.com/questions/72167844/estimation-fiber-length-of-overlapping-fibers-from-image-using-python-code

from scipy import ndimage
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage import morphology
from skan.csr import skeleton_to_csgraph
from skan import draw, Skeleton, summarize

img00 = cv2.imread(r'xlinkimage.jpg')
img_01 = cv2.cvtColor(img00, cv2.COLOR_BGR2GRAY)
img0 = cv2.cvtColor(img00, cv2.COLOR_BGR2GRAY)

i_size = min(np.size(img_01,1),600) # image size for imshow
# Creating kernel
kernel = np.ones((2, 2), np.uint8)
  
# Using cv2.dialate() method 
img01 = cv2.dilate(img0, kernel, iterations=2)
cv2.imwrite('Img1_Filtered.jpg',img01)

ret,thresh1 = cv2.threshold(img01,245,255,cv2.THRESH_BINARY)
thresh = (thresh1/255).astype(np.uint8)
cv2.imwrite('Img2_Binary.jpg',thresh1)

# skeleton based on Lee's method
skeleton1 = (skeletonize(thresh, method='lee')/255).astype(bool)
skeleton1 = morphology.remove_small_objects(skeleton1, 100, connectivity=2)

# fiber Detection through skeletonization and its characterization
SPACING_NM = 1   # pixel

fig, ax = plt.subplots()
draw.overlay_skeleton_2d(img_01, skeleton1, dilate=1, axes=ax)

pixel_graph, coordinates0 = skeleton_to_csgraph(skeleton1, spacing=SPACING_NM)

skel_analysis = Skeleton(skeleton1, spacing=SPACING_NM,source_image=img00)
branch_data = summarize(skel_analysis)
branch_data.hist(column='branch-distance', bins=100)
draw.overlay_euclidean_skeleton_2d(img_01, branch_data,skeleton_color_source='branch-type')

dd = ndimage.distance_transform_edt(thresh)
radii = np.multiply(dd, skeleton1)
Fiber_D_mean = np.mean(2*radii[radii>0])
criteria = 2 * Fiber_D_mean # Remove branches smaller than this length for characterization

dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
_,mv,_,mp = cv2.minMaxLoc(dist)

aa = branch_data[(branch_data['branch-distance']>criteria)]
CNT_L_count, CNT_L_mean, CNT_L_stdev = aa['branch-distance'].describe().loc[['count','mean','std']]
print("Fiber Length (px[enter image description here][1])  : Count, Average, Stdev:",int(CNT_L_count),round(CNT_L_mean,2),round(CNT_L_stdev,2))
#print(branch_data)
print(pixel_graph)
summa = np.sum(branch_data)
#print(dist)
print("Antal vita pixlar")
print(np.sum(thresh1 == 255))
print("Antal Svarta Pixlar")
print(np.sum(thresh1 == 0))
print("Total längd")
print(summa[3])
print("genomsnitt bred")
print("In Pixels: ",np.sum(thresh ==1) / summa [3], " and in Nm: " , np.sum(thresh ==1) / summa [3] * 2.78)
print(mv*2, mp) #mv längsta sträckan från bakgrund vilket borde ge radien på fibern
cv2.imshow('Distance Transform Image', dist)
cv2.imshow('Skeleton', img_01)
cv2.waitKey(0)