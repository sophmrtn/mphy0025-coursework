# Script to load in data and display CT images
# Author: Sophie Martin

# import required modules
import os
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from python_code_for_coursework_part1 import utilsCoursework

ATLAS_DIR = './head_and_neck_images/atlas'
TUNE_DIR  =  './head_and_neck_images/tune'
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

# Get all files in atlas and tune folders
atlas_files = os.listdir(ATLAS_DIR)
tune_files = os.listdir(TUNE_DIR)

# load images from atlas and tune folders
atlas_imgs = [load_image(os.path.join(ATLAS_DIR, f)) for f in atlas_files]
tune_imgs = [load_image(os.path.join(TUNE_DIR, f)) for f in tune_files]
print(atlas_files)

# use dispImage utility function to view the CT images
utilsCoursework.dispImage(load_image('./head_and_neck_images/atlas/atlas_1.png'))

# overlay contours
plt.contour(np.transpose(load_image('./head_and_neck_images/atlas/atlas_1_BRAIN_STEM.png')),colors='lime', linewidths=0.5)
plt.contour(np.transpose(load_image('./head_and_neck_images/atlas/atlas_1_SPINAL_CORD.png')), colors='red',  linewidths=0.5)
plt.show()



