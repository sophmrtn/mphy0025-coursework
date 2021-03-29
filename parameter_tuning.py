# Script that tunes the registration parameters
# Author: Sophie Martin

# import required modules and functions
from data import load_image, visualise_image, ATLAS_DIR, TUNE_DIR
from utilsCoursework import dispImage, resampImageWithDefField
from demonsReg import demonsReg
import matplotlib.pyplot as plt
import os
import numpy as np

for i in range(1,4):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(3,2, 1)
    ax.set_title('Target CT Image')
    # Display CT image for the tuning patient
    visualise_image(i, TUNE_DIR, ax=ax)
    tune_img = load_image(os.path.join(TUNE_DIR, ('tune_%d.png' % (i))))
    
    # Loop over the 5 atlas patients and register to the tuning image as a target
    for j in range(1,6):
        atlas_img = load_image(os.path.join(ATLAS_DIR, ('atlas_%d.png' % (j))))

        # Use demonsReg to perform registration, atlas as source, tuning img as the target
        warped_atlas, def_field = demonsReg(atlas_img, tune_img, disp_freq=0)

        # Warp the spinal cord and brain stem images using the deformation field ensuring that they remain as binary image
        brain_stem = load_image(os.path.join(ATLAS_DIR, ('atlas_%d_BRAIN_STEM.png' % (j))))
        spinal_cord = load_image(os.path.join(ATLAS_DIR, ('atlas_%d_SPINAL_CORD.png' % (j))))

        warped_brain_stem = resampImageWithDefField(brain_stem, def_field, interp_method='nearest', pad_value=0)
        warped_spinal_cord = resampImageWithDefField(spinal_cord, def_field, interp_method='nearest', pad_value=0)

        # Check warped images are also binary (0 and 1 (255))
        if any(np.unique(warped_brain_stem) != np.array([0, 255])) or any(np.unique(warped_spinal_cord) != np.array([0, 255])):
            print('Check - warped images are not binary')

        # Display warped CT image with warped contours overlayed
        ax = fig.add_subplot(3,2, j+1)
        ax.set_title('ATLAS Patient %d' % j)

        # use dispImage utility function to plot warped atlas
        dispImage(warped_atlas, ax=ax)

        # overlay warped contours: requires taking the transpose due to matplot lib contour function
        plt.contour(np.transpose(warped_brain_stem),colors='lime', linewidths=0.5)
        plt.contour(np.transpose(warped_spinal_cord), colors='red',  linewidths=0.5)

    plt.show()