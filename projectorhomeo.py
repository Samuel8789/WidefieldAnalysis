# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:43:04 2024

@author: sp3660
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the coordinates in the source image (rectangle_image)
source_coords= np.array([
    [539, 409],  # Top-left corner
    [539, 309],  # Top-right corner
    [739, 309],  # Bottom-right corner
    [739, 409]   # Bottom-left corner
], dtype='float32')
full_frame = np.array([
    [0, 700],  # Top-left corner
    [0, 0],  # Top-right corner
    [1200, 0],  # Bottom-right corner
    [1200, 700]   # Bottom-left corner
], dtype='float32')
# Define the coordinates in the destination image (reference_img)
dest_coords = proj_mask_full_fov.astype('float32')
rect = cv2.minAreaRect(dest_coords)
box = cv2.boxPoints(rect)
# Compute the homography matrix
H, _ = cv2.findHomography(source_coords, box)

# Print the homography matrix
print("Homography Matrix:")
print(H)

# Optional: Visualize the transformation on an image
# Create images for visualization
rect_img = np.zeros((720, 1280, 3), dtype=np.uint8)  # Rectangle image size
ref_img = np.zeros((1080, 1280, 3), dtype=np.uint8)  # Reference image size

psych = rect_img
fov = ref_img
# Draw the rectangle in the source image
cv2.polylines(rect_img, [np.int32(source_coords)], isClosed=True, color=(0, 255, 0), thickness=2)

# Draw the rectangle in the destination image
cv2.polylines(ref_img, [np.int32(box)], isClosed=True, color=(0, 255, 0), thickness=2)

# Apply the homography to the source rectangle
transformed_coords = cv2.perspectiveTransform(np.array([source_coords]), H)[0]

# Draw the transformed rectangle in the reference image
cv2.polylines(ref_img, [np.int32(transformed_coords)], isClosed=True, color=(255, 0, 0), thickness=2)

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Source Image")
plt.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Destination Image with Transformed ROI")
plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()


# Define the new ROI coordinates in the destination image
new_roi_coords_dest = reregistered_gabor_mask_full_fov.astype('float32').reshape(-1, 1, 2)
new_roi_coords_dest_gabor=registered_map_mask_full_fov.astype('float32').reshape(-1, 1, 2)

# Compute the inverse of the homography matrix
H_inv = np.linalg.inv(H)

# Apply the inverse homography to the new ROI coordinates
new_roi_coords_src = cv2.perspectiveTransform(new_roi_coords_dest, H_inv)
new_roi_coords_src_gabor = cv2.perspectiveTransform(new_roi_coords_dest_gabor, H_inv)


# Print the new coordinates in the source image
print("Transformed Coordinates in Source Image:")
print(new_roi_coords_src)


# Draw the new ROI in the destination image
cv2.polylines(fov, [np.int32(new_roi_coords_dest)], isClosed=True, color=(0, 255, 0), thickness=2)
cv2.polylines(fov, [np.int32(new_roi_coords_dest_gabor)], isClosed=True, color=(0, 255, 255), thickness=2)


# Draw the transformed ROI in the source image
cv2.polylines(psych, [np.int32(new_roi_coords_src)], isClosed=True, color=(255, 255, 0), thickness=2)
cv2.polylines(psych, [np.int32(new_roi_coords_src_gabor)], isClosed=True, color=(0, 255, 255), thickness=2)


# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Destination Image with New ROI")
plt.imshow(cv2.cvtColor(psych, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Source Image with Transformed ROI")
plt.imshow(cv2.cvtColor(fov, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

#%% create image to project
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the size of the image
width, height = 1280, 720
# Create a blank (black) image
binary_mask = np.zeros((height, width), dtype=np.uint8)

# Define the coordinates of the polygon (for example purposes)

polygon_coords=new_roi_coords_src.squeeze().astype('int32')


# Draw the polygon on the image
cv2.polylines(binary_mask, [polygon_coords], isClosed=True, color=255, thickness=2)

# Fill the polygon (if needed, to make the mask solid)
cv2.fillPoly(binary_mask, [polygon_coords], color=255)

# Display the mask
plt.figure(figsize=(8, 6))
plt.title("Binary Mask")
plt.imshow(binary_mask, cmap='gray')
plt.axis('off')
plt.show()

# Save the mask as a JPEG file (optional)
cv2.imwrite('binary_mask_v1.jpg', binary_mask)

binary_mask_gabor = np.zeros((height, width), dtype=np.uint8)

# Define the coordinates of the polygon (for example purposes)

polygon_coords_gabor=new_roi_coords_src_gabor.squeeze().astype('int32')

# Draw the polygon on the image
cv2.polylines(binary_mask_gabor, [polygon_coords_gabor], isClosed=True, color=255, thickness=2)

# Fill the polygon (if needed, to make the mask solid)
cv2.fillPoly(binary_mask_gabor, [polygon_coords_gabor], color=255)


# Save the mask as a JPEG file (optional)
cv2.imwrite('binary_mask_gabor.jpg', binary_mask_gabor)
#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the larger and smaller masks
# Ensure these masks are loaded as binary images (0s and 255s)
larger_mask = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Scale\binary_mask_gabor.jpg', cv2.IMREAD_GRAYSCALE)
smaller_mask = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Scale\binary_mask_v1.jpg', cv2.IMREAD_GRAYSCALE)

# Check the sizes of the masks
print(f"Larger mask size: {larger_mask.shape}")
print(f"Smaller mask size: {smaller_mask.shape}")

# Ensure the smaller mask is the same size as the larger mask (if needed, you can resize or pad)
# In this case, we assume the smaller mask is already correctly positioned within the larger mask

# Create a new mask where the smaller mask is subtracted from the larger mask
# Subtract the smaller mask from the larger mask
new_mask = cv2.subtract(larger_mask, smaller_mask)

# Display the masks
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Larger Mask")
plt.imshow(larger_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Smaller Mask")
plt.imshow(smaller_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("New Mask (Larger - Smaller)")
plt.imshow(new_mask, cmap='gray')
plt.axis('off')

plt.show()

# Save the new mask as a JPEG file (optional)
cv2.imwrite('all_v1_but_gabor.jpg', new_mask)
