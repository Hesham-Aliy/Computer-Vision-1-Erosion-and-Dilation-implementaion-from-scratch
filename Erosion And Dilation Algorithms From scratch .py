#!/usr/bin/env python
# coding: utf-8

# # <font style = "color:rgb(50,120,229)">Implementation of Morphological Operations</font>
# 

# ## <font style="color:rgb(50,120,229)">Implement Method 2</font>
# 1. Scan through the image and superimpose the kernel on the neighborhood of each pixel. 
# 1. Perform an AND operation of the neighborhood with the kernel.
# 1. Replace the pixel value with the `maximum` value in the neighborhood given by the kernel. 
# 
# This means that you check every pixel and its neighborhood with respect to the kernel and change the pixel to white if any of the pixel in this neighborhood is white. OpenCV implements an optimized version of this method. This will work even if the image is not a binary image.

# ## <font style="color:rgb(50,120,229)">Import Libraries </font>

# In[1]:


import cv2
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


# # <font style="color:rgb(50,120,229)">Create a Demo Image</font>
# ## <font style="color:rgb(50,120,229)">Create an empty matrix </font>

# In[3]:


im = np.zeros((10,10),dtype='uint8')
print(im);
plt.imshow(im)


# ## <font style="color:rgb(50,120,229)">Lets add some white blobs</font>
# 
# We have added the blobs at different places so that all boundary cases are covered in this example.

# In[4]:


im[0,1] = 1
im[-1,0]= 1
im[-2,-1]=1
im[2,2] = 1
im[5:8,5:8] = 1

print(im)
plt.imshow(im)


# This becomes our demo Image for illustration purpose

# ## <font style="color:rgb(50,120,229)">Create an Ellipse Structuring Element </font>
# Let us create a 3x3 ellipse structuring element.

# In[5]:


element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
print(element)


# In[6]:


ksize = element.shape[0]


# In[7]:


height,width = im.shape[:2]


# ## <font style="color:rgb(50,120,229)">First check the correct output using cv2.dilate</font>

# In[8]:


dilatedEllipseKernel = cv2.dilate(im, element)
print(dilatedEllipseKernel)
plt.imshow(dilatedEllipseKernel)


# ## <font style="color:rgb(50,120,229)">Now Code for Dilation from scratch</font>
# 
# We will Create a VideoWriter object and write the result obtained at the end of each iteration to the object. Save the video to **`dilationScratch.avi`** and display it using markdown below:
# 
# **`dilationScratch.avi` will come here**
# 
# ```<video width="320" height="240" controls>
#   <source src="dilationScratch.avi" type="video/mp4">
# </video>```
# 
# **Note**
# 
# 1. we will Use FPS as 10 and frame size as 50x50
# 2. Before writing the frame, resize it to 50x50
# 3. Convert the resized frame to BGR
# 4. Release the object

# In[9]:


border = ksize//2
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
paddedDilatedIm = paddedIm.copy()

border = ksize//2
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
paddedDilatedIm = paddedIm.copy()
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video=cv2.VideoWriter("dilationScratch.avi",fourcc,10,(50,50))
for h_i in range(border, height+border):
    for w_i in range(border,width+border):
        # When you find a white pixel
        if im[h_i-border,w_i-border]:
            print("White Pixel Found @ {},{}".format(h_i,w_i))
            
            paddedDilatedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1] =                 cv2.bitwise_or(paddedDilatedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1],element)
            
            print(paddedDilatedIm)
            plt.imshow(paddedDilatedIm);plt.show()
            
            #resizing
            resized_frame=cv2.resize(paddedDilatedIm,(50,50))
            resized_frame=resized_frame*255
            resized_frame=cv2.cvtColor(resized_frame,cv2.COLOR_BAYER_GB2BGR)
            video.write(resized_frame)

video.release()


# In[10]:


dilatedImage = paddedDilatedIm[border:border+height,border:border+width]
plt.imshow(dilatedImage)


# # <font style="color:rgb(50,120,229)">Implement Erosion </font>

# ## <font style="color:rgb(50,120,229)">Check the correct output using cv2.erode </font>

# In[11]:


ErodedEllipseKernel = cv2.erode(im, element)
print(ErodedEllipseKernel)
plt.imshow(ErodedEllipseKernel);


# ## <font style="color:rgb(50,120,229)">Now code for Erosion from scratch</font>
# 
# Create a VideoWriter object and write the result obtained at the end of each iteration to the object. Save the video to **`erosionScratch.avi`** and display it using markdown below:
# 
# **`erosionScratch.avi` will come here**
# 
# ```<video width="320" height="240" controls>
#   <source src="erosionScratch.avi" type="video/mp4">
# </video>```
# 
# **Note**
# 
# 1. We will Use FPS as 10 and frame size as 50x50
# 2. Before writing the frame, resize it to 50x50
# 3. Convert the resized frame to BGR
# 4. Release the object

# In[12]:


border = ksize//2
paddedIm = np.zeros((height + border*2, width + border*2))
paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 1)
paddedErodedIm = paddedIm.copy()
paddedErodedIm2= paddedIm.copy()

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video2=cv2.VideoWriter("erosionScratch.avi",fourcc,10,(50,50))
roi=0
temp=0
for h_i in range(border, height+border):
    for w_i in range(border,width+border):
        if im[h_i-border,w_i-border]:
            roi=paddedErodedIm2[h_i-border  : (h_i + border)+1, w_i - border : (w_i + border)+1] 
            temp= cv2.bitwise_or(roi,element)
            paddedErodedIm[h_i,w_i]=np.min(temp)
            
            print(paddedErodedIm)
            plt.imshow(paddedErodedIm);plt.show()
            resized_frame2=cv2.resize(paddedErodedIm,(50,50))
            resized_frame2=resized_frame2*255
            resized_frame2=cv2.cvtColor(resized_frame2,cv2.COLOR_BAYER_GB2BGR)
            video.write(resized_frame2)

video.release()


# In[13]:


erodedImage = paddedErodedIm[border:border+height,border:border+width]
plt.imshow(erodedImage)


# In[ ]:




