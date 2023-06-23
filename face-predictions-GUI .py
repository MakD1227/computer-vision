#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
 
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)
import joblib
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
import pickle 

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})faces in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
    
     # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            #print("The folder where currently seen is: ", subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)
   
#Below, we define the RGB2GrayTransformer and HOGTransformer.

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
  
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
     def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

#all set to preprocess our RGB images to scaled HOG features.
def run(include='A'):
    data_path = fr'face'
    include=include
    base_name = 'recognition'
    width=150
    resize_all(src=data_path, pklname=base_name,width=width,include=include)
    data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
    X = np.array(data['data'])
    
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(
        pixels_per_cell=(14, 14), 
        cells_per_block=(2,2), 
        orientations=9, 
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()

    # call fit_transform on each transform converting X_train step by step
    X_gray = grayify.fit_transform(X)
    X_hog = hogify.fit_transform(X_gray)
    X_prepared = scalify.fit_transform(X_hog)
    # load the model from disk
    filename='faceregonition.sav'
    model = pickle.load(open(filename, 'rb'))
    x=model.predict(X_prepared )
    return x[0]
    #print("This habitant is : ",)
  
#importing neccessary libraries
from tkinter import Tk,Label,Button,Text
import pickle
import pandas as pd

def  here(event):
    
    text=input_text.get('1.0','end')
    x=run(text)
    if(x=='unkown'):
        response='locked'
    else:
        respose='unlock'
  
    #intializing output Text object
    text2=Text(root,height=2,bg="pink",padx=5,pady=5)
    
    text2.insert('1.0','The door is '+response)
    text2['state']='disabled'
    text2.grid(row=3,column=1)
root=Tk()
#root.iconbitmap('security.ico')
root.title("door unlock using  AI secuirty camera")
background='lightcyan'
root.configure(bg=background)
#adding label to promot the user to enter text
Label(root,text="Enter here=>",bg="lightgreen",padx=10,pady=5).grid(row=0,column=0)

#adding Text object (used to enter text) to the root 
input_text=Text(root,height=2,bg="lightyellow",padx=5,pady=5)
input_text.insert('1.0','')
input_text.grid(row=0,column=1)
 
#adding "go" button
go_button=Button(root,bg="red",text="go",padx=10,pady=5,)
go_button.bind('<Button-1>', here)
go_button.grid(row=0,column=2,)

#add nothing to do label to have space between the first row and the second row
Label(root,text="",bg=background).grid(row=1,column=0)

#adding  "Output label"
Label(root,text="Output",bg="lightblue",padx=10,pady=5).grid(row=2,column=0)

root.geometry("800x300")
root.mainloop()

