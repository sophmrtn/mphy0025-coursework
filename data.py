# Script to load in data and display CT images
# Author: Sophie Martin

# import required modules
import os
import matplotlib.pyplot as plt
import skimage.io
import numpy as np

# Load data from the ATLAS and tuning patients with contours

def load_image(filepath):
    '''
    Function to load png image from filepath, convert to double and reorientate the images for 
    proper displaying using dispImage utility function.
    
    Inputs: filepath - path to desired image
    Outputs: img - returns image object
    '''
    img = skimage.io.imread(filepath)
    img = np.double(img)
    img = np.flipud(img)
    # take transpose to switch x and y
    img = np.transpose(img)
    return img




