import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

#defining width of picture to show in window
#note the analysis is done on full size picture anyway
basewidth = 900

#Create window with theme and size
root = ttk.Window(themename="superhero")
root.title("Analyse Fibers in picture")
root.geometry('1000x1000')

def loadPicture():
    global img
    global img_orginal
    #Get filename to load into opencv
    root.filename = filedialog.askopenfilename(initialdir="./Images/",title="Select File", filetypes=(("jpg","*.jpg"),("png","*.png")))
    #Open file as opencv file
    img_orginal = cv2.imread(root.filename)
    
    label_filename.configure(text=root.filename)
    
    #Convert opencv file to Tkinter pillow format and resize
    #Just for me to test really
    #basewidth = 500
    img = img_orginal
    img_fromarray = Image.fromarray(img)
    width_percent = (basewidth/float(img_fromarray.size[0]))
    hsize = int((float(img_fromarray.size[1])*float(width_percent)))
    img_fromarray = img_fromarray.resize((basewidth, hsize), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(image= img_fromarray)
    
    #Change the label to the loaded picture
    label.img_tk = img_tk
    label.configure(image=img_tk)
    
    pass

def applyInvert():
    global img
    
    img = cv2.bitwise_not(img,img)
    
    #Convert opencv file to Tkinter pillow format
    #Just for me to test really
    #basewidth = 500
    img_fromarray = Image.fromarray(img)
    width_percent = (basewidth/float(img_fromarray.size[0]))
    hsize = int((float(img_fromarray.size[1])*float(width_percent)))
    img_fromarray = img_fromarray.resize((basewidth, hsize), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(image= img_fromarray)
    
    #Change the label to the loaded picture
    label.img_tk = img_tk
    label.configure(image=img_tk)
    
    pass


def applyThresh():
    global img
    global img_orginal
    
    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.cvtColor(img_orginal, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, float(lowerThreshValue.get()), float(upperThreshValue.get()), cv2.THRESH_BINARY)
    #dilated = cv2.dilate(thresh, kernel, iterations=int(it_dilate.get()))
    
    img = thresh
    #Convert opencv file to Tkinter pillow format
    #Just for me to test really
    #basewidth = 500
    img_fromarray = Image.fromarray(img)
    width_percent = (basewidth/float(img_fromarray.size[0]))
    hsize = int((float(img_fromarray.size[1])*float(width_percent)))
    img_fromarray = img_fromarray.resize((basewidth, hsize), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(image= img_fromarray)
    
    #Change the label to the loaded picture
    label.img_tk = img_tk
    label.configure(image=img_tk)
    pass
def applyDilate():
    global img
    kernel = np.ones((2, 2), np.uint8)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(img, kernel, iterations=int(it_dilate.get()))
    
    img = dilated
    #Convert opencv file to Tkinter pillow format
    #Just for me to test really
    #basewidth = 500
    img_fromarray = Image.fromarray(img)
    width_percent = (basewidth/float(img_fromarray.size[0]))
    hsize = int((float(img_fromarray.size[1])*float(width_percent)))
    img_fromarray = img_fromarray.resize((basewidth, hsize), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(image= img_fromarray)
    
    #Change the label to the loaded picture
    label.img_tk = img_tk
    label.configure(image=img_tk)
    
    
    pass


def performAnalysis():
    global img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255,45,45)
    text_thickness = 2
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    MIN = 200

    #masken = np.zeros(img.shape, dtype="uint8")
    img_temp = cv2.imread(root.filename)
    for contour in contours:
        #img_temp = cv2.imread('xlinkimage.jpg')
        if len(contour) > MIN:
            #masken = np.zeros(img.shape, np.uint8)
            masken = np.zeros_like(img)
            cv2.drawContours(masken, [contour], -1 , (255),-1)
            cv2.drawContours(img_temp, [contour], -1 , (255),2)
            result = cv2.distanceTransform(masken, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            result2 = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            text_pos = (max_loc[0] + 50, max_loc[1] - 50)
            img_temp = cv2.putText(img_temp, str(round((max_val*2)*2.78, 2)),text_pos, font, 1 , text_color,text_thickness, cv2.LINE_AA )
            cv2.circle(img_temp, max_loc , 3,(0,255,0),1)
            cv2.line(img_temp,text_pos,max_loc,text_color,2)
            cv2.imshow('Distance transform', img_temp)
    cv2.imshow("Result",result)
    
    
    pass
def updatedLowerThresh(value_thresh):
    value_thresh = round(float(value_thresh),0)
    lowerThreshValue.set(str(value_thresh))
    #applyThresh()
    pass
def updatedupperThresh(value_thresh):
    value_thresh = round(float(value_thresh),0)
    upperThreshValue.set(str(value_thresh))
    #applyThresh()
    pass
def imageToMemory():
    global img_memory
    
    img_memory = img
    
    pass
def showMemory():
    cv2.imshow("Memorized picture",img_memory)
    pass

def addMemory():
    global img
    global img_memory
    
    alpha = 0.5
    beta = (1.0 - alpha)
        
    img = cv2.addWeighted(img,alpha,img_memory,beta,0.0)
        
    #Convert opencv file to Tkinter pillow format
    #Just for me to test really
    #basewidth = 500
    img_fromarray = Image.fromarray(img)
    width_percent = (basewidth/float(img_fromarray.size[0]))
    hsize = int((float(img_fromarray.size[1])*float(width_percent)))
    img_fromarray = img_fromarray.resize((basewidth, hsize), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(image= img_fromarray)
    
    #Change the label to the loaded picture
    label.img_tk = img_tk
    label.configure(image=img_tk)
    
    pass

lowerThreshValue = tk.StringVar()
upperThreshValue = tk.StringVar()
#Create empty picture as placeholder until a file is loaded
img_empty = np.zeros([600,900,3],dtype=np.uint8)
img_fromarray = Image.fromarray(img_empty)
img_tk = ImageTk.PhotoImage(image= img_fromarray)

#Label intended for showing picture once it is loaded
label = tk.Label(text="Please load file! No File is loaded!", justify=LEFT, padx=0)
label.grid(row=0,column=0, columnspan=5)
label.img_tk = img_tk
label.configure(image=img_tk)

#Open filedialog and load a file into OpenCV
b1 = ttk.Button(root, text="Load picture", bootstyle=SUCCESS, command=loadPicture, width=15)
b1.grid(row=1,column=0)

#Label for filname once a file i choosen
label_filename = ttk.Label(text="No file choosen!", anchor='e')
label_filename.grid(row=1,column=1, columnspan=4)

#Button to run Threshold process
b2 = ttk.Button(root, text="Apply Threshold!", bootstyle=(INFO, OUTLINE),command=applyThresh, width=15)
b2.grid(row=2,column=0, padx=5)

#Slider for threshold
lowerThresh = ttk.Scale(orient=HORIZONTAL, from_=10, to=255, length=150, command=updatedLowerThresh )
lowerThresh.grid(row=2, column=1)

#Textbox for lower Threshold
lowerThreshEntry = ttk.Entry(textvariable=lowerThreshValue,width=5)
lowerThreshEntry.grid(row=2,column=2)

#Slider for threshold
upperThresh = ttk.Scale(orient=HORIZONTAL, from_=10, to=255, length=150, command=updatedupperThresh )
upperThresh.grid(row=2, column=3)

#Textbox for upper Threshold
upperThreshEntry = ttk.Entry(textvariable=upperThreshValue,width=5)
upperThreshEntry.grid(row=2,column=4)

#Button to run Dilate process
b3 = ttk.Button(root, text="Dilate!", bootstyle=(INFO, OUTLINE),command=applyDilate, width=15)
b3.grid(row=3,column=0)

#Button to run Analysis process
b4 = ttk.Button(root, text="Analyse!", bootstyle=(INFO, OUTLINE),command=performAnalysis, width=15)
b4.grid(row=4,column=0)

#Spinbox for number of iterations of dilation
it_dilate = ttk.Spinbox(bootstyle="danger", from_=1,to=10, width=3, justify='left')
it_dilate.set(3)
it_dilate.grid(row=3,column=1)

#Button to run Invert process
b5 = ttk.Button(root, text="Invert!", bootstyle=(INFO, OUTLINE),command=applyInvert, width=15)
b5.grid(row=5,column=0)

#Button to run put image in memory
b6 = ttk.Button(root, text=" -> Image B!", bootstyle=(INFO, OUTLINE),command=imageToMemory, width=15)
b6.grid(row=5,column=1)

#Button to show picture in memory
b7 = ttk.Button(root, text="Show Image B!", bootstyle=(INFO, OUTLINE),command=showMemory, width=15)
b7.grid(row=5,column=2)

#Button to show picture in memory
b8 = ttk.Button(root, text="Add B to current!", bootstyle=(INFO, OUTLINE),command=addMemory, width=15)
b8.grid(row=5,column=3)



root.mainloop()