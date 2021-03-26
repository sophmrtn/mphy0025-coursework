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


def visualise_image(img_no, directory=ATLAS_DIR, ax=None):
    '''
    Function to plot image and contours for a given image number and path to the directory.

    Inputs: (int) img_no - integer
    directory - path to directory containing indexed images

    Outputs: none - displays image
    '''
    # get basename for filepaths based on directory
    basename = os.path.basename(directory)

    # get filepaths for images and masks using known patten
    img_filepath = os.path.join(directory, ('%s_%d.png' % (basename, img_no)))
    brainstem_mask_filepath = os.path.join(directory, ('%s_%d_BRAIN_STEM.png' % (basename, img_no)))
    spinal_mask_filepath = os.path.join(directory, ('%s_%d_SPINAL_CORD.png' % (basename, img_no)))

    # use dispImage utility function to view the CT images
    utilsCoursework.dispImage(load_image(img_filepath), ax=ax)

    # overlay contours: requires taking the transpose due to matplot lib contour function
    plt.contour(np.transpose(load_image(brainstem_mask_filepath)),colors='lime', linewidths=0.5)
    plt.contour(np.transpose(load_image(spinal_mask_filepath)), colors='red',  linewidths=0.5)

fig = plt.figure(figsize=(10,8))
for i in range(1,6):
    ax = fig.add_subplot(3,2, i)
    visualise_image(i, ATLAS_DIR, ax=ax)
    ax.set_title('Patient: %d' % i)

plt.suptitle('ATLAS')
plt.tight_layout()

fig = plt.figure(figsize=(10,8))
for i in range(1,4):
    ax = fig.add_subplot(3,2, i)
    visualise_image(i, TUNE_DIR, ax=ax)
    ax.set_title('Patient: %d' % i)

plt.suptitle('TUNE')
plt.tight_layout()
plt.show()