# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:08:50 2025

@author: sp3660
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from skimage import io, feature, morphology
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from scipy.interpolate import interp2d, LinearNDInterpolator
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import skimage as ski
from datetime import datetime
import pickle

def load_image(path, grayscale=False):
    """
    Load an image from the specified path if the file exists.
    
    Parameters:
    path (str or Path): The file path of the image.
    grayscale (bool): If True, converts the image to grayscale.
    
    Returns:
    np.array or None: The loaded image or None if the file is missing.
    """
    path = Path(path)
    if path.exists():
        image = cv2.imread(str(path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if grayscale else image
    else:
        print(f"Warning: File not found - {path}")
        return None

def load_mat_file(path, key):
    """
    Load a .mat file from the specified path if it exists and extract the specified key.
    
    Parameters:
    path (str or Path): The file path of the .mat file.
    key (str): The key to extract from the .mat file.
    
    Returns:
    np.array or None: The extracted data or None if the file is missing or the key is not found.
    """
    path = Path(path)
    if path.exists():
        data = loadmat(str(path))
        return data.get(key, None)
    else:
        print(f"Warning: File not found - {path}")
        return None

# Define the main image folder where all data is stored
image_folder = Path(r'C:/Users/sp3660/Documents/Projects/LabNY/Amsterdam/Analysis/TempMovies/Animal3p020')

# Load images and store them in a dictionary for easy access
images = {
    "gabor_reference": load_image(image_folder / 'WholeBrain_GaborMappingDay.jpg', grayscale=True),
    "gabor_roi": load_image(image_folder / 'ROI_GaborMappingDay.jpg', grayscale=True),
    "gabor_mask1": load_image(image_folder / 'gabor_mask_Inner.png', grayscale=True),
    "gabor_mask2": load_image(image_folder / 'gabor_mask_Middle.png', grayscale=True),
    "gabor_mask3": load_image(image_folder / 'gabor_mask_Outer.png', grayscale=True),
    "map_reference": load_image(image_folder / 'WholeBrain_RetinotopicMappingDay.jpg', grayscale=True),
    "map_roi": load_image(image_folder / 'ROI_RetinotopicMappingDay.jpg', grayscale=True),
    "map_mask": load_image(image_folder / 'retinotopy_mask.png', grayscale=True),
    "proj_ref": load_image(image_folder / 'WholeBrain_ExperimentDay.jpg', grayscale=True),
    "grid_image": load_image(image_folder / 'Grid_snapshot.jpg')
}

# Load grid data from a .mat file
grid_data = load_mat_file(image_folder / 'grid_2025130.mat', 'fullgrid')

# Close any open plots
plt.close('all')

# Create a figure with multiple subplots for visualization
f, ax = plt.subplots(3, 2, figsize=(10, 10))
f.tight_layout()

# Display images in the subplots
ax.flatten()[0].imshow(images["gabor_reference"], cmap='gray')
ax.flatten()[1].imshow(images["map_reference"], cmap='gray')
ax.flatten()[2].imshow(images["gabor_roi"], cmap='gray')
ax.flatten()[3].imshow(images["map_roi"], cmap='gray')
ax.flatten()[5].imshow(images["map_mask"], cmap='gray')

# Ensure that gabor masks exist before stacking them
valid_gabor_masks = [img for img in [images["gabor_mask1"], images["gabor_mask2"], images["gabor_mask3"]] if img is not None]
if valid_gabor_masks:
    stacked_gabor_masks = np.sum(np.stack(valid_gabor_masks, axis=0), axis=0)
    ax.flatten()[4].imshow(stacked_gabor_masks, cmap='gray')
else:
    print("Warning: No valid Gabor masks found for display.")

ax.flatten()[5].imshow(images["map_mask"], cmap='gray')

# Show the figure
plt.show()
#%%