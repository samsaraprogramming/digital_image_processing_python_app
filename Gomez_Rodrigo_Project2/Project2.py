# Author: Rodrigo Gomez
# Date: May 4, 2020


# import the necessary packages
import PIL
import tkinter as tk
import tkinter.filedialog as tkFileDialog
import cv2
import numpy as np
import copy
import time 
from math import sqrt,exp

from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
from tkinter import messagebox
from tkinter.ttk import *

from PIL import Image
from PIL import ImageTk

#Important image variables
original_image = None
rotate_image = None
save_image = None

#Image panels for all windows 
panelA = None
panelB = None
panelC = None
panelD = None


#FrequencyFilterChoice
freq_filter_chosen = False



def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


#Ideal Filters
def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) <= D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) <= D0:
                base[y,x] = 0
    return base
    
def idealFilterBandstop(D1, D2,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D1:
                base[y,x] = 1
            elif distance((y,x),center) > D2:
                base[y,x] = 1
            
    return base
    
def idealFilterBandpass(D1, D2,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D1:
                base[y,x] = 0
            elif distance((y,x),center) > D2:
                base[y,x] = 0
            
    return base
    
   
#Butterworth Filters   
def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base
    

def butterworthBandpass(D1, D2 ,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    band_center = (D1 + D2)/2
    width= abs(D2-D1)
    
    for x in range(cols):
        for y in range(rows):
            if((distance((y,x),center)**2-band_center**2) != 0):
                base[y,x] = 1 - (1/(1+(distance((y,x),center)*width/(distance((y,x),center)**2-band_center**2))**(2*n)))
            else:
                base[y,x] = 1
    return base

def butterworthBandstop(D1, D2 ,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    band_center = (D1 + D2)/2
    width= abs(D2-D1)
    
    for x in range(cols):
        for y in range(rows):
            if((distance((y,x),center)**2-band_center**2) != 0):
                base[y,x] = 1/(1+(distance((y,x),center)*width/(distance((y,x),center)**2-band_center**2))**(2*n))
            else:
                base[y,x] = 0
    return base


#Gaussian Filters
def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianBandstop(D1, D2, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    band_center = (D1 + D2)/2
    width = abs(D1-D2)
    
    for x in range(cols):
        for y in range(rows):
            if(distance((y,x),center) !=0):
                base[y,x] = 1 - exp(-(((distance((y,x),center)**2 - band_center**2)/(distance((y,x),center)*width))**2))
            else:
                base[y,x] = 1
    return base
    
def gaussianBandpass(D1, D2, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    band_center = (D1 + D2)/2
    width = abs(D1-D2)
        
    for x in range(cols):
        for y in range(rows):
            if(distance((y,x),center) !=0):
                base[y,x] = 1 - (1 - exp(-(((distance((y,x),center)**2 - band_center**2)/(distance((y,x),center)*width))**2)))
            else:
                base[y,x] = 0
    return base
    
def gaussianHomomorphic(D0, imgShape, high_frequency_gain, low_frequency_gain):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = (high_frequency_gain- low_frequency_gain)*(1 - exp(((-distance((y,x),center)**2)/(D0**2)))) + low_frequency_gain
    return base




#Perform User-specified function
def perform_function():
    global root, root1, root2, root3
    global combo
    global panelA, panelB, panelC, panelD
    global original_image, numpy_image, rotate_image, save_image
    global u, v, Entry3, Entry4, Entry5, Entry6
    global freq_filter_chosen
    
    
    selection = combo.get()
    selection = selection[0]
    
    
    
    #Grey Block
    #Create Medium Grey Block
    if(selection == 'A'):
        original_image = None
        root1.iconify()
        root2.iconify()
        
        numpy_image = np.zeros([256,256],dtype=np.uint8)
        numpy_image.fill(125) # or img[:] = 255
        save_image = copy.deepcopy(numpy_image)
        #Convert Numpy Array to PIL format
        numpy_image_PIL = PIL.Image.fromarray(numpy_image)
        #Convert from PIL format to ImageTk format
        numpy_image_ImageTk = PIL.ImageTk.PhotoImage(numpy_image_PIL)
        
        panelA.configure(image=numpy_image_ImageTk)
        panelA.image = numpy_image_ImageTk
        panelB.configure(image=numpy_image_ImageTk)
        panelB.image = numpy_image_ImageTk
        root1.withdraw()
        root2.withdraw()
        root.deiconify()
    
    #Fill rows with same value as row number - 1
    if(selection == 'B'):   
        original_image = None
        root1.iconify()
        root2.iconify()
        
        numpy_image = np.zeros([256,256],dtype=np.uint8)
        for x in range(256):
            numpy_image[x,:] = x
        
        save_image = copy.deepcopy(numpy_image)
        
        #Convert Numpy Array to PIL format
        numpy_image_PIL = PIL.Image.fromarray(numpy_image)
        #Convert from PIL format to ImageTk format
        numpy_image_ImageTk = PIL.ImageTk.PhotoImage(numpy_image_PIL)
        
        panelA.configure(image=numpy_image_ImageTk)
        panelA.image = numpy_image_ImageTk
        panelB.configure(image=numpy_image_ImageTk)
        panelB.image = numpy_image_ImageTk
        
        root1.withdraw()
        root2.withdraw()
        root.deiconify()
    
        
    #Fill columns with same value as column number - 1
    if(selection == 'C'):
        original_image = None
        root1.iconify()
        root2.iconify()
        
        numpy_image = np.zeros([256,256],dtype=np.uint8)
        for x in range(256):
            numpy_image[:,x] = x
        
        save_image = copy.deepcopy(numpy_image)
        #Convert Numpy Array to PIL format
        numpy_image_PIL = PIL.Image.fromarray(numpy_image)
        #Convert from PIL format to ImageTk format
        numpy_image_ImageTk = PIL.ImageTk.PhotoImage(numpy_image_PIL)   
        
        panelA.configure(image=numpy_image_ImageTk)
        panelA.image = numpy_image_ImageTk
        panelB.configure(image=numpy_image_ImageTk)
        panelB.image = numpy_image_ImageTk
        
        root1.withdraw()
        root2.withdraw()
        root.deiconify()
        
    #Change Brightness by adding or subtracting a constant
    if(selection == 'D'):
        #Check if an image has been selected
        if original_image is not None:
            #Prompt User for Brightness Value to Add/Subtract
            combo.config(state='disabled')
            value = simpledialog.askinteger("Input Brightness Value", "What brightness value to add/subtract?", parent = root1)
            #Check input was not cancelled
            if(value is None):
               combo.config(state='readonly')
               return
            
            alpha = 1 #Constant to multiple pixel values
            beta = value #Constant to add to pixel values
            image_copy = cv2.addWeighted(original_image, alpha, np.zeros(original_image.shape, original_image.dtype), 0, beta)
            save_image = copy.deepcopy(image_copy)
            #Convert OpenCV image colorspace to PIL image colorspace
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy = PIL.ImageTk.PhotoImage(image_copy)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy)
                panelD.image = image_copy
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=image_copy)
                panelB.image = image_copy  
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Change contrast by multiplying pixel values by a constant 
    if(selection == 'E'):
        #Check if an image has been selected
        if original_image is not None:
            combo.config(state='disabled')
            #Prompt User for Contrast value to multiply
            value = simpledialog.askfloat("Input Contrast Value", "What contrast value to multiply? Enter a positive real number.", parent = root1)
            #Check input was not cancelled
            if(value is None):
               combo.config(state='readonly')
               return
            
            alpha = value
            beta = 0
            image_copy = cv2.addWeighted(original_image, alpha, np.zeros(original_image.shape, original_image.dtype), 0, beta)
            save_image = copy.deepcopy(image_copy)
            #Convert OpenCV image colorspace to PIL image colorspace
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy = PIL.ImageTk.PhotoImage(image_copy)
            
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy)
                panelD.image = image_copy
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=image_copy)
                panelB.image = image_copy 
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Change contrast by raising pixel values to a specified exponent value
    if(selection == 'F'):
        if original_image is not None:
            combo.config(state='disabled')
            #Prompt User for Contrast Exponential Value
            power = simpledialog.askfloat("Input Contrast Power Value", "What contrast power value would you like to try? Enter a positive real number.", parent = root1)
            #Check input was not cancelled
            if(power is None):
               combo.config(state='readonly')
               return
          
            image_copy = np.array(255*(original_image/255)**power, dtype = 'uint8')
            save_image = copy.deepcopy(image_copy) 
            #Convert OpenCV image colorspace to PIL image colorspace
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy = PIL.ImageTk.PhotoImage(image_copy)
            
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy)
                panelD.image = image_copy
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=image_copy)
                panelB.image = image_copy
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Resize Image to user specified size
    if(selection == 'G'):
        if original_image is not None:
            combo.config(state='disabled')
            #Prompt User for Desired Width and Height
            width = simpledialog.askinteger("Input Width Size", "What width would you like the image? Current image size is: Width=" + str(original_image.shape[1]) + ", Height=" + str(original_image.shape[0]) + ".", parent = root1)
            if(width is None):
               combo.config(state="readonly")
               return
            height = simpledialog.askinteger("Input Height Size", "What height would you like the image? Current image size is: Width=" + str(original_image.shape[1]) + ", Height=" + str(original_image.shape[0]) + ".", parent = root1)
            if(height is None):
               combo.config(state="readonly")
               return
            
            
            new_size = (width, height)
            image_copy = cv2.resize(original_image, new_size, interpolation = cv2.INTER_AREA)
            save_image = copy.deepcopy(image_copy)
            #Convert OpenCV image colorspace to PIL image colorspace
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy = PIL.ImageTk.PhotoImage(image_copy)
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy)
                panelD.image = image_copy
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=image_copy)
                panelB.image = image_copy
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
                    
            
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Rotate image in 90 degree increments
    if(selection == 'H'):
        if original_image is not None:
            if rotate_image is None:
                rotate_image = copy.deepcopy(original_image)
            #Prompt User for Brightness Value to Add/Subtract
            rotate_image = cv2.rotate(rotate_image, cv2.ROTATE_90_CLOCKWISE)
            save_image = copy.deepcopy(rotate_image)
            rotate_image_brg = cv2.cvtColor(rotate_image,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            rotate_image_PIL = PIL.Image.fromarray(rotate_image_brg)
            #Convert from PIL format to ImageTk format
            rotate_image_ImageTk = PIL.ImageTk.PhotoImage(rotate_image_PIL)
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=rotate_image_ImageTk)
                panelD.image = rotate_image_ImageTk
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=rotate_image_ImageTk)
                panelB.image = rotate_image_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Flip image horizontally
    if(selection == 'I'):
        if original_image is not None:
            flipped_image =  cv2.flip(original_image, 1)
            save_image = copy.deepcopy(flipped_image)
            flipped_image = cv2.cvtColor(flipped_image,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            flipped_image_PIL = PIL.Image.fromarray(flipped_image)
            #Convert from PIL format to ImageTk format
            flipped_image_ImageTk = PIL.ImageTk.PhotoImage(flipped_image_PIL)
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=flipped_image_ImageTk)
                panelD.image = flipped_image_ImageTk
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=flipped_image_ImageTk)
                panelB.image = flipped_image_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()   
           
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
            
    #Flip image vertically
    if(selection == 'J'):
        if original_image is not None:
            flipped_image =  cv2.flip(original_image, 0)
            save_image = copy.deepcopy(flipped_image)
            flipped_image = cv2.cvtColor(flipped_image,cv2.COLOR_BGR2RGB)
            
            #Convert Numpy Array to PIL format
            flipped_image_PIL = PIL.Image.fromarray(flipped_image)
            #Convert from PIL format to ImageTk format
            flipped_image_ImageTk = PIL.ImageTk.PhotoImage(flipped_image_PIL)
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=flipped_image_ImageTk)
                panelD.image = flipped_image_ImageTk
                root1.deiconify()
                root2.deiconify()
            else:
                panelB.configure(image=flipped_image_ImageTk)
                panelB.image = flipped_image_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()   
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
        
    #Spatial Low-pass Filter
    if(selection == 'K'):
        if original_image is not None:
            combo.config(state='disabled')
            #Prompt User for Desired Kernel Size
            kernel_size = simpledialog.askstring("Input Kernel Size", "What kernel size would you like to try? Available size: 3x3, 5x5, 7x7, 9x9. Use input format as 3x3, 5x5, etc.", parent = root1)
            if(kernel_size is None):
               combo.config(state='readonly')
               return
            elif(kernel_size != '3x3' and kernel_size != '5x5' and kernel_size != '7x7' and kernel_size != '9x9'):
                messagebox.showerror("Invalid Kernel_size","Please enter a valid kernel_size.")
                combo.config(state='readonly')
                return
                
            
            kernel_size.split('x')
            kernel_width = int(kernel_size.split('x')[0])
            kernel_height = int(kernel_size.split('x')[1])
            kernel = np.ones((kernel_width, kernel_height), np.float32)/(kernel_width*kernel_height)
            
            image_copy =  cv2.filter2D(original_image, -1, kernel)
            save_image = copy.deepcopy(image_copy)
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy_PIL = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy_ImageTk)
                panelD.image = image_copy_ImageTk
                
                
                root.deiconify()
                root1.deiconify()
                root2.deiconify()
                
            else:
                panelB.configure(image=image_copy_ImageTk)
                panelB.image = image_copy_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
  
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Spatial High-pass Filter
    if(selection == 'L'):
        if original_image is not None:
            
            kernel = np.ones((3, 3), np.float32)
            kernel = np.negative(kernel)
            kernel[int(3/2), int(3/2)] = 8
            
            image_copy =  cv2.filter2D(original_image, -1, kernel)
            save_image = copy.deepcopy(image_copy)
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy_PIL = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy_ImageTk)
                panelD.image = image_copy_ImageTk
                
                
                root.deiconify()
                root1.deiconify()
                root2.deiconify()
                
            else:
                panelB.configure(image=image_copy_ImageTk)
                panelB.image = image_copy_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
  
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Spatial High-boost Filter
    if(selection == 'M'):
        if original_image is not None:
            combo.config(state='disabled')
            #Prompt User for Desired Height and WidthSize
            kernel_size = simpledialog.askstring("Input Kernel Size", "What kernel size would you like to try? Available size: 3x3, 5x5, 7x7, 9x9. Use input format as 3x3, 5x5, etc.", parent = root1)
            if(kernel_size is None):
               combo.config(state='readonly')
               return
            elif(kernel_size != '3x3' and kernel_size != '5x5' and kernel_size != '7x7' and kernel_size != '9x9'):
                messagebox.showerror("Invalid Kernel_size","Please enter a valid kernel_size.")
                combo.config(state='readonly')
                return
                
            coefficient_value = simpledialog.askfloat("Input  A-Coefficient Value", "What coefficient would you like to try? Input a real positive number.", parent = root1)
            if(coefficient_value is None):
               combo.config(state='readonly')
               return
            elif(coefficient_value <= 0):
                messagebox.showerror("Invalid coefficient","Please enter a real positive number.")
                combo.config(state='readonly')
                return
          
            kernel_size.split('x')
            kernel_width = int(kernel_size.split('x')[0])
            kernel_height = int(kernel_size.split('x')[1])
            kernel = np.ones((kernel_width, kernel_height), np.float32)/(kernel_width*kernel_height)
           
            image_copy =  cv2.filter2D(original_image, -1, kernel)
            
            image_copy = cv2.addWeighted(original_image, coefficient_value, image_copy  , -1, 0)
            
            save_image = copy.deepcopy(image_copy)
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            
            #Convert Numpy Array to PIL format
            image_copy_PIL = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy_ImageTk)
                panelD.image = image_copy_ImageTk
                
                
                root.deiconify()
                root1.deiconify()
                root2.deiconify()
                
            else:
                panelB.configure(image=image_copy_ImageTk)
                panelB.image = image_copy_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
  
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
     #Global Histogram
    if(selection == 'N'):
        if original_image is not None:
            
            image_copy =  cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
            #equalize the histogram of the Y channel
            image_copy[:,:,0] = cv2. equalizeHist(image_copy[:,:,0])
            # convert YUV image to RGB
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_YUV2BGR)
            
            save_image = copy.deepcopy(image_copy)
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy_PIL = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy_ImageTk)
                panelD.image = image_copy_ImageTk
                
                
                root.deiconify()
                root1.deiconify()
                root2.deiconify()
                
            else:
                panelB.configure(image=image_copy_ImageTk)
                panelB.image = image_copy_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
  
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Adaptive Histogram
    if(selection == 'O'):
        if original_image is not None:
            
            image_copy =  cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
            
            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #equalize the histogram of the Y channel
            image_copy[:,:,0] = clahe.apply(image_copy[:,:,0])
            # convert YUV image to RGB
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_YUV2BGR)
            
            
            
            save_image = copy.deepcopy(image_copy)
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy_PIL = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy_ImageTk)
                panelD.image = image_copy_ImageTk
                
                
                root.deiconify()
                root1.deiconify()
                root2.deiconify()
                
            else:
                panelB.configure(image=image_copy_ImageTk)
                panelB.image = image_copy_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
  
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    #Frequency Domain Filters
    if(selection == 'P'):
        if original_image is not None:
            
            if(freq_filter_chosen == False):
                combo.config(state='disabled')
                root3.deiconify()
                return
            else:
                #Get Frequency Filter Details
                Filter_Type = u.get()
                Frequency_Range = v.get()
                First_Cutoff_Frequency_Radius = Entry3.get()
                Second_Cutoff_Frequency_Radius = Entry4.get()
                Butterworth_Order = Entry5.get()
                High_Boost_Factor = Entry6.get()
                
                u.set("I")
                v.set("L")
                Entry3.delete(0, 'end')
                Entry4.delete(0, 'end')
                Entry4.config(state = 'disabled')
                Entry5.delete(0, 'end')
                Entry5.config(state = 'disabled')
                Entry6.delete(0, 'end')
                Entry6.config(state = 'disabled')
                
                
                if((Filter_Type != "G" and Filter_Type != "I" and Filter_Type != "B")):
                    combo.config(state='readonly')
                    freq_filter_chosen = False
                    return
                    
                   
                
                if((Frequency_Range != "L" and Frequency_Range != "H" and Frequency_Range != "HB" and Frequency_Range != "BP" and Frequency_Range != "BS")):
                    combo.config(state='readonly')
                    freq_filter_chosen = False
                    return
                
                
                
                
                if(First_Cutoff_Frequency_Radius.isdigit() == True):
                    First_Cutoff_Frequency_Radius = int(First_Cutoff_Frequency_Radius)
                else:
                    messagebox.showerror("Invalid Cutoff Frequency Radius","Please enter a positive integer cutoff frequency radius.")
                    combo.config(state='readonly')
                    freq_filter_chosen = False
                    return
                
                if((Frequency_Range == "BP" or Frequency_Range == "BS") and Second_Cutoff_Frequency_Radius.isdigit() == True):
                
                    Second_Cutoff_Frequency_Radius = int(Second_Cutoff_Frequency_Radius)
                    if Second_Cutoff_Frequency_Radius <= First_Cutoff_Frequency_Radius:
                        messagebox.showerror("Invalid Cutoff Frequency Radius","Please enter a second cutoff frequency radius greater than the first.")
                        combo.config(state='readonly')
                        freq_filter_chosen = False
                        return
                elif(Frequency_Range == "BP" or Frequency_Range == "BS"):
                    messagebox.showerror("Invalid Cutoff Frequency Radius","Please enter a positive integer cutoff frequency radius.")
                    combo.config(state='readonly')
                    freq_filter_chosen = False
                    return
                   
                
                if(Frequency_Range == "HB"):
                    try:
                        High_Boost_Factor = float(High_Boost_Factor)
                    except ValueError:
                       messagebox.showerror("Invalid High-Boost-Factor","Please enter a positive decimal value for the high-boost factor.")
                       combo.config(state='readonly')
                       freq_filter_chosen = False
                       return
                    if(High_Boost_Factor <= 0):
                        messagebox.showerror("Invalid High-Boost-Factor","Please enter a positive decimal value for the high-boost factor.")
                        combo.config(state='readonly')
                        freq_filter_chosen = False
                        return
                
                    
                    
                   
                if(Filter_Type == "B" and Butterworth_Order.isdigit() == True): 
                    Butterworth_Order = int(Butterworth_Order)  
                    
                    
                elif(Filter_Type == "B"):
                    messagebox.showerror("Invalid Butterwort Order","Please enter a positive integer value for the Butterworth Order.")
                    combo.config(state='readonly')
                    freq_filter_chosen = False
                    return
                
                
                image_copy =  cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
                original_shape = image_copy[:,:,0]
                
                original_shape = original_shape.astype(np.float_)
                original_fft = np.fft.fft2(original_shape)
                center = np.fft.fftshift(original_fft)
                
                
                if(Filter_Type == "I" and Frequency_Range == "H"):   
                    center_filtered = center * idealFilterHP(First_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "I" and Frequency_Range == "L"):
                    center_filtered = center * idealFilterLP(First_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "I" and Frequency_Range == "BP"):
                    center_filtered = center * idealFilterBandpass(First_Cutoff_Frequency_Radius, Second_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "I" and Frequency_Range == "BS"):
                    center_filtered = center * idealFilterBandstop(First_Cutoff_Frequency_Radius, Second_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "I" and Frequency_Range == "HB"):
                    center_filtered = (High_Boost_Factor - 1) *center + center * idealFilterHP(First_Cutoff_Frequency_Radius, original_shape.shape)
                    
                
                elif(Filter_Type == "G" and Frequency_Range == "H"):
                    center_filtered = center * gaussianHP(First_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "G" and Frequency_Range == "L"):
                    center_filtered = center * gaussianLP(First_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "G" and Frequency_Range == "BS"):
                    center_filtered = center * gaussianBandstop(First_Cutoff_Frequency_Radius, Second_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "G" and Frequency_Range == "BP"):
                    center_filtered = center * gaussianBandpass(First_Cutoff_Frequency_Radius, Second_Cutoff_Frequency_Radius, original_shape.shape)
                elif(Filter_Type == "G" and Frequency_Range == "HB"):
                    center_filtered = (High_Boost_Factor - 1) * center + center * gaussianHP(First_Cutoff_Frequency_Radius, original_shape.shape)
                
                elif(Filter_Type == "B" and Frequency_Range == "H"):
                    center_filtered = center * butterworthHP(First_Cutoff_Frequency_Radius, original_shape.shape, Butterworth_Order)
                elif(Filter_Type == "B" and Frequency_Range == "L"):
                    center_filtered = center * butterworthLP(First_Cutoff_Frequency_Radius, original_shape.shape, Butterworth_Order)
                elif(Filter_Type == "B" and (Frequency_Range == "BP")):
                    center_filtered = center * butterworthBandpass(First_Cutoff_Frequency_Radius, Second_Cutoff_Frequency_Radius, original_shape.shape, Butterworth_Order)
                elif(Filter_Type == "B" and (Frequency_Range == "BS")):
                    center_filtered = center * butterworthBandstop(First_Cutoff_Frequency_Radius, Second_Cutoff_Frequency_Radius, original_shape.shape, Butterworth_Order)
                elif(Filter_Type == "B" and Frequency_Range == "HB"):
                    center_filtered = (High_Boost_Factor - 1) * center + center * butterworthHP(First_Cutoff_Frequency_Radius, original_shape.shape, Butterworth_Order)
                    
                filtered_image = np.fft.ifftshift(center_filtered)
               
                inversed_fft_image = np.fft.ifft2(filtered_image)
                real_image = np.abs(inversed_fft_image)
                real_image = np.clip(real_image, 0.0, 255.0)
                real_image = real_image.astype(np.uint8)
                
                image_copy[:,:,0] = real_image
                image_copy =  cv2.cvtColor(image_copy, cv2.COLOR_YUV2BGR)
                
                save_image = copy.deepcopy(image_copy)
                image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
                #Convert Numpy Array to PIL format
                image_copy_PIL = PIL.Image.fromarray(image_copy)
                #Convert from PIL format to ImageTk format
                image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
               
                if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                    panelD.configure(image=image_copy_ImageTk)
                    panelD.image = image_copy_ImageTk
                    
                    
                    root.deiconify()
                    root1.deiconify()
                    root2.deiconify()
                    
                else:
                    panelB.configure(image=image_copy_ImageTk)
                    panelB.image = image_copy_ImageTk
                    root1.withdraw()
                    root2.withdraw()
                    root.deiconify()
                freq_filter_chosen = False
      
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    
    #Homomorphic Filter
    if(selection == 'Q'):
        if original_image is not None:
            combo.config(state='disabled')
            #Prompt User for Filter Details
            cutoff_freq_radius1 = simpledialog.askinteger("Input First Cutoff Frequency Radius", "What cutoff frequency radius would you like to try? Input a positive integer value", parent = root1)
            
            if cutoff_freq_radius1 is None:
               combo.config(state='readonly')
               return
            elif (cutoff_freq_radius1 < 0):
               messagebox.showerror("Invalid Cutoff Frequency Radius","Please enter a valid cutoff frequency radius.")
               combo.config(state='readonly')
               return
               
            high_frequency_gain = simpledialog.askfloat("Input High Frequency Gain", "What high frequency gain would you like to try? Enter a positive real number.", parent = root1)
            
            if high_frequency_gain is None:
               combo.config(state='readonly')
               return
            elif (high_frequency_gain <= 0):
               messagebox.showerror("Invalid High Frequency Gain","Please enter a valid high frequency gain.")
               combo.config(state='readonly')
               return 

            low_frequency_gain = simpledialog.askfloat("Input Low Frequency Gain", "What low frequency gain would you like to try? Enter a positive real number.", parent = root1)
            
            if low_frequency_gain is None:
               combo.config(state='readonly')
               return 
               
            elif (low_frequency_gain <= 0):
               messagebox.showerror("Invalid Low Frequency Gain","Please enter a valid low frequency gain.")
               combo.config(state='enabled')
               return
          
            
            image_copy =  cv2.cvtColor(original_image, cv2.COLOR_BGR2YUV)
            original_shape = image_copy[:,:,0]
            original_shape = original_shape.astype(np.float_)
            original_fft = np.fft.fft2(original_shape)
            center = np.fft.fftshift(original_fft)
            
            
            center_filtered = center * gaussianHomomorphic(cutoff_freq_radius1, original_shape.shape, high_frequency_gain, low_frequency_gain)
                
            filtered_image = np.fft.ifftshift(center_filtered)
           
            inversed_fft_image = np.fft.ifft2(filtered_image)
            real_image = np.abs(inversed_fft_image)
            real_image = np.clip(real_image, 0.0, 255.0)
            real_image = real_image.astype(np.uint8)
            
            
            image_copy[:,:,0] = real_image
            image_copy =  cv2.cvtColor(image_copy, cv2.COLOR_YUV2BGR)
            
        
            save_image = copy.deepcopy(image_copy)
            image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
            #Convert Numpy Array to PIL format
            image_copy_PIL = PIL.Image.fromarray(image_copy)
            #Convert from PIL format to ImageTk format
            image_copy_ImageTk = PIL.ImageTk.PhotoImage(image_copy_PIL)
           
            if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
                panelD.configure(image=image_copy_ImageTk)
                panelD.image = image_copy_ImageTk
                
                root.iconify
                root.deiconify()
                root1.deiconify()
                root2.deiconify()
                
            else:
                panelB.configure(image=image_copy_ImageTk)
                panelB.image = image_copy_ImageTk
                root1.withdraw()
                root2.withdraw()
                root.deiconify()
  
        else:
            messagebox.showerror("No Image Error","Please select an image to perform function on.")
    
    if(selection == 'S'):
        messagebox.showerror("No Function Error", "Please select function to perform first.")
        
    combo.config(state='readonly')
    





#Save Image 
def save_image_function(*args):
    global root
    global save_image
    if(save_image is not None):
        files = [('Jpeg File', '*.jpg'),
                ('All Files', '*.*')] 
        
        path = tkFileDialog.asksaveasfilename(title = "Select file",filetypes = files, defaultextension = files)
        if not path:
            pass
        else:
            cv2.imwrite(path, save_image)
    else:
        messagebox.showerror("No Modified Image Error", "Please select an image and perform operation before saving.")
    







#Select Image
def select_image():
    # grab a reference to important global variables
    global original_image, save_image, rotate_image
    global panelA, panelB, panelC, panelD
    global root, root1, root2
    global numpy_image_ImageTk

    save_image = None
    rotate_image = None
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename(title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

    # ensure a file path was selected
    if len(path) > 0:
        
        original_image = cv2.imread(path)
        image = original_image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format
        image = Image.fromarray(image)
        

        # convert to ImageTk format
        image = ImageTk.PhotoImage(image)
        
        
        if(original_image.shape[0] > 256 or original_image.shape[1] > 256):
            if panelA is not None or panelB is not None:
                panelA.configure(image=numpy_image_ImageTk)
                panelB.configure(image=numpy_image_ImageTk)
                panelA.image = numpy_image_ImageTk
                panelB.image = numpy_image_ImageTk
            else:
                pass
            if panelC is None or panelD is None:
                # the first panel will store our original image
                
                panelC = Label(root1, image=image)
                panelC.image = image
                panelC.pack()

                # while the second panel will store the edge map
                panelD = Label(root2, image=image)
                panelD.image = image
                panelD.pack()

            # otherwise, update the image panels
            else:
                panelC.configure(image=image)
                panelD.configure(image=image)
                panelC.image = image
                panelD.image = image
                
            root.deiconify()
            root1.deiconify()
            root2.deiconify()
            
            
            
        # if the panels are None, initialize them
        else:
            if panelA is None or panelB is None:
                # the first panel will store our original image
                panelA = Label(root, image=image)
                panelA.image = image
                panelA.pack(side="left", padx=10, pady=10)

                # while the second panel will store the edge map
                panelB = Label(root, image=image)
                panelB.image = image
                panelB.pack(side="right", padx=10, pady=10)

            # otherwise, update the image panels
            else:
                # update the pannels
                panelA.configure(image=image)
                panelB.configure(image=image)
                panelA.image = image
                panelB.image = image
                
            root1.withdraw()
            root2.withdraw()
            root.deiconify()
            

def perform_freq_functions():
    global freq_filter_chosen
    freq_filter_chosen = True
    root3.withdraw()
    perform_function()
    
def activate_entry4():
    global Entry4, Entry6
    Entry4.config(state='enabled')
    Entry6.delete(0, 'end')
    Entry6.config(state='disabled')
    
    
def activate_entry5():
    global Entry5
    Entry5.config(state='enabled')
def deactivate_entry5():
    global Entry5
    Entry5.delete(0, 'end')
    Entry5.config(state='disabled')
    
    
def activate_entry6():
    global Entry4, Entry6
    Entry6.config(state='enabled')
    Entry4.delete(0, 'end')
    Entry4.config(state='disabled')
    
    
def deactivate_entry4_entry6():
    global Entry4, Entry6
    Entry4.delete(0, 'end')
    Entry4.config(state='disabled')
    Entry6.delete(0, 'end')
    Entry6.config(state='disabled')
    
def withdraw_freq_filter_window():
    combo.config(state='readonly')
    root3.withdraw()
    
    



# Main window initialization
root = Tk()
#root.iconbitmap(default='Camera.ico')
root.wm_title("Digital Image Processing Application")
root.geometry('800x500')
positionRight = int(root.winfo_screenwidth()/2 - 800/2)
positionDown = int(root.winfo_screenheight()/2 - 500/2)
root.geometry("+{}+{}".format(positionRight, positionDown))


#Create Numpy Array of White Pixels of size 256x256 px 
#For Default Image on Main Window
numpy_image = np.zeros([256,256],dtype=np.uint8)
numpy_image.fill(255)
#Convert Numpy Array to PIL format
numpy_image_PIL = PIL.Image.fromarray(numpy_image)
#Convert from PIL format to ImageTk format
numpy_image_ImageTk = PIL.ImageTk.PhotoImage(numpy_image_PIL)


#Create image panels for main window
panelA = tk.Label(root, image=numpy_image_ImageTk, borderwidth= 2, relief ="sunken", width = 256, height = 256)
panelA.image = numpy_image_ImageTk
panelA.place(x=80,y=40)
panelB = tk.Label(root, image=numpy_image_ImageTk, borderwidth= 2, relief ="sunken", width = 256, height = 256)
panelB.image = numpy_image_ImageTk
panelB.place(x=464,y=40)


#create combobox for main window
combo = Combobox(root, width = 40, state='readonly')
combo['values'] = ("Select a function", "A) Create Medium Grey Block", "B) Row Pixels = Row Index - 1", "C) Row Pixels = Column Index - 1", 
                   "D) Adjust Brightness", "E) Adjust Contrast by Multiplication", "F) Adjust Contrast by Power", "G) Resize Image", 
                   "H) Rotate 90 degrees", "I) Horizontal Mirror", "J) Vertical Mirror", "K) Spatial Low-Pass Filter", "L)Spatial High-Pass Filter", 
                   "M) Spatial High-boost Filter", "N) Global Histogram Equalization", "O) Adaptive Histogram Equalization", "P) Frequency-based Filters",
                   "Q) Homomorphic Filter")
combo.current(0)
combo.place(x=80,y=350)


btn_load = ttk.Button(root, text="Select an image", width=25, command=  select_image)
btn_load.place(x=80,y=400)
btn_update = ttk.Button(root, text="Update image", width=25, command= perform_function)
btn_update.place(x=240,y=400)
btn_save = ttk.Button(root, text="Save image", width=25, command=save_image_function)
btn_save.place(x=400,y=400)
btn_quit = ttk.Button(root, text='Quit Program', width=25, command=root.destroy)
btn_quit.place(x=560,y=400)

#Before Image Window Initialization
root1 = Toplevel(root)
root1.wm_title("Before Image Processing")
root1.protocol("WM_DELETE_WINDOW", root1.withdraw)
positionRight = int(root1.winfo_screenwidth()/4 - 800/2 + 150)
positionDown = int(root1.winfo_screenheight()/2 - 600/2)
root1.geometry("+{}+{}".format(positionRight, positionDown))
root1.withdraw()

#After Image Window Initialization
root2 = Toplevel(root)
root2.wm_title("After Image Processing")
root2.protocol("WM_DELETE_WINDOW", root2.withdraw)
positionRight = int(root2.winfo_screenwidth()/4 + 800/2)
positionDown = int(root2.winfo_screenheight()/2 - 600/2)
root2.geometry("+{}+{}".format(positionRight, positionDown))
root2.withdraw()

#Frequency Filters Choice window 
root3 = Toplevel(root)
root3.wm_title("Frequency Filter Settings")
root3.geometry('400x300')
root3.protocol("WM_DELETE_WINDOW", withdraw_freq_filter_window)
positionRight = int(root3.winfo_screenwidth()/2 - 800/2)
positionDown = int(root3.winfo_screenheight()/2 - 500/2)
root3.geometry("+{}+{}".format(positionRight, positionDown))
root3.withdraw()


MODES = [
        ("Ideal", "I"),
        ("Gaussian", "G"),
        ("Butterworth", "B"),
    ]

u = StringVar()
u.set("I") # initialize

Label1 = Label(root3, text='Filter Types', font=('Calibri', 10))
Label1.grid(row=0,column=0, sticky='w', padx = 10)


count = 1
for text, mode in MODES:
    if(text == 'Butterworth'):
        b = Radiobutton(root3, text=text,
                       variable=u, value=mode, command=activate_entry5)
    else:
        b = Radiobutton(root3, text=text,
                   variable=u, value=mode, command=deactivate_entry5)
    b.grid(row =count, column=0, sticky = 'w', padx=10)
    count += 1
    
MODES = [
        ("Low-Pass", "L"),
        ("High-Pass", "H"),
        ("High-Boost", "HB"),
        ("Band-Pass", "BP"),
        ("Band-Stop", "BS"),
    ]

v = StringVar()
v.set("L") # initialize

Label2 = Label(root3, text='Frequency Range', font=('Calibri', 10))
Label2.grid(row=0,column=1, sticky='w', padx = 10)

count = 1
for text, mode in MODES:
    if(text == "High-Boost"):
        b = Radiobutton(root3, text=text,
                       variable=v, value=mode, command=activate_entry6)
    elif(text == "Band-Pass" or text == "Band-Stop"):
        b = Radiobutton(root3, text=text,
                       variable=v, value=mode, command=activate_entry4)
    else:
        b = Radiobutton(root3, text=text,
                       variable=v, value=mode, command=deactivate_entry4_entry6)
                       
    b.grid(row =count, column=1, sticky ='w', padx=10)
    count += 1
    
Label3 = Label(root3, text='First Cutoff Frequency Radius', font=('Calibri', 10))
Label3.grid(row=6,column=0, padx = 5, pady=5, sticky='w')
Entry3 = Entry(root3, width=30)
Entry3.grid(row=6,column=1, padx = 5)

Label4 = Label(root3, text='Second Cutoff Frequency Radius', font=('Calibri', 10))
Label4.grid(row=7,column=0, padx = 5, pady = 5, sticky='w')
Entry4 = Entry(root3, width=30)
Entry4.grid(row=7,column=1, padx = 5)
Entry4.config(state='disabled')

Label5 = Label(root3, text='Butterworth Order', font=('Calibri', 10))
Label5.grid(row=8,column=0, padx = 5, pady = 5, sticky='w')
Entry5 = Entry(root3, width=30)
Entry5.grid(row=8,column=1, padx = 5)
Entry5.config(state='disabled')

Label6 = Label(root3, text='High-Boost Factor', font=('Calibri', 10))
Label6.grid(row=9,column=0, padx = 5, pady = 5, sticky='w')
Entry6 = Entry(root3, width=30)
Entry6.grid(row=9,column=1, padx = 5)
Entry6.config(state='disabled')

btn_exit = ttk.Button(root3, text="Exit", width=25, command=  withdraw_freq_filter_window)
btn_exit.grid(row=11, column=0, padx=5, pady=15)
btn_enter = ttk.Button(root3, text="Enter", width=25, command=  perform_freq_functions)
btn_enter.grid(row=11, column=1, padx=5, pady=15)




# kick off the GUI
root.mainloop()