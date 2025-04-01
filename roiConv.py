import numpy as np
import nibabel as nib
from scipy.io import loadmat

# Load the MarsBaR .mat file
marsbar_data = loadmat('./roi/Amygdala_L_1_roi.mat')

# Extract the ROI data
roi_data = marsbar_data['roi']  # Adjust the key based on your .mat file structure

# Create a NIFTI image
# You'll need to know the dimensions and affine transformation of your original image
img = nib.Nifti1Image(roi_data, affine=np.eye(4))

# Save the NIFTI file
nib.save(img, 'Amygdala_L1.nii')