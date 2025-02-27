# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:19:00 2025

@author: sp3660
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io, feature, morphology
import cv2
from scipy.spatial import distance
from skimage import io, measure
from scipy.interpolate import interp2d
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
# Load .mat file containing the original grid
image_folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Animal3p020'


# PATH MANAGING
gabor_reference = cv2.imread(image_folder+r'\WholeBrain_GaborMappingDay.jpg')




grid_file = image_folder+r'\grid_2025130.mat'
data = loadmat(grid_file)
A = data['fullgrid']  # Extract the array from the .mat file

# Load the .tif image (distorted projection of the grid)
grid_image_file = image_folder+r'\Grid_snapshot.jpg'
grid_image = io.imread(grid_image_file)

# Step 1: Detect edges in the .tif image using Canny edge detector
edges = feature.canny(grid_image, sigma=3)

# Step 2: Post-process the edges to fill small gaps (dilate and close)
edges_dilated = morphology.dilation(edges, morphology.square(2))  # Dilate edges to close small gaps
edges_closed = morphology.closing(edges_dilated, morphology.square(2))  # Close small gaps between edges

# Convert the edges to an 8-bit image (required for OpenCV)
edges_8bit = (edges_closed * 255).astype(np.uint8)

# Step 3: Find contours using OpenCV
contours, hierarchy = cv2.findContours(edges_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Filter to keep only external contours (those with no parent)
external_contours = []
for i, h in enumerate(hierarchy[0]):  # Hierarchy format: [Next, Previous, First Child, Parent]
    if h[3] == -1:  # No parent means it's an external contour
        external_contours.append(contours[i])

# Step 5: Calculate the center of each external contour
contour_centers = []
for contour in external_contours:
    contour = contour.squeeze()  # Remove unnecessary dimensions for processing
    M = cv2.moments(contour)
    if M['m00'] != 0:  # Avoid division by zero
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        contour_centers.append([cX, cY])

coordinates =np.array(contour_centers)
# Step 1: Sort the points based on their y-coordinate
sorted_by_y = coordinates[coordinates[:, 0].argsort()]

# Step 2: Group the points by their y-coordinate (rows)
# Assuming that rows are defined by having the same or similar y-values
# We'll cluster points into rows where the difference in y-values is small
rows = []
current_row = [sorted_by_y[0]]

for i in range(1, len(sorted_by_y)):
    if abs(sorted_by_y[i, 0] - sorted_by_y[i - 1, 0]) < 10:  # Adjust threshold as needed
        current_row.append(sorted_by_y[i])
    else:
        rows.append(np.array(current_row))
        current_row = [sorted_by_y[i]]

# Don't forget to add the last row
rows.append(np.array(current_row))

# Step 3: Sort each row by x-coordinate
sorted_rows = [row[row[:, 1].argsort()] for row in rows]

# Step 4: Combine the sorted rows back into a single array
sorted_coordinates = np.vstack(sorted_rows)

# Print the result
print(sorted_coordinates)

# Step 6: Visualize the external contours with their centers

# Step 7: Define the centers of the squares in the original grid
label_image = measure.label(A, connectivity=2)  # Label connected components
regions = measure.regionprops(label_image)
# Extract the centers of the detected squares

square_centers = np.array([region.centroid for region in regions])


max_y = np.max(square_centers[:, 0])  # Find the maximum y-value
filtered_centers = square_centers[square_centers[:, 0] < max_y]  # Remove points with max y
# Step 8: Ensure that the number of contour centers matches the number of squares
# This may involve sorting or otherwise aligning the centers, depending on your needs

sorted_indices = np.lexsort((filtered_centers[:, 0], filtered_centers[:, 1]))  # Sort by column 0, then column 1
sorted_centers = filtered_centers[sorted_indices]
sorted_centers[:, [0, 1]] = sorted_centers[:, [1, 0]]

f,ax=plt.subplots(1,2)
ax[0].imshow(grid_image, cmap='gray')
for contour in external_contours:
    contour = contour.squeeze()
    ax[0].plot(contour[:, 0], contour[:, 1], color='r', linewidth=1)

# Plot contour centers
ax[0].scatter(sorted_coordinates[:, 0], sorted_coordinates[:, 1], color='blue', s=10, label="Contour Centers")
ax[1].imshow(A, cmap='gray')
ax[1].scatter(square_centers[:, 1], square_centers[:, 0], color='red', s=10, label="Detected Centers")

#%%
x=np.unique(sorted_centers[:, 0])
y= np.unique(sorted_centers[:, 1])
X, Y = np.meshgrid(np.unique(sorted_centers[:, 0]), np.unique(sorted_centers[:, 1]))
dat = sorted_coordinates.reshape(15, 7, 2)

from scipy.interpolate import RegularGridInterpolator
interpx = RegularGridInterpolator((x, y), dat[:,:,0],bounds_error=False,fill_value=None)
interpy = RegularGridInterpolator((x, y), dat[:,:,1],bounds_error=False,fill_value=None)

new_coordinate=[550,632]

boundsx=[sorted_coordinates[:,0].min(),sorted_coordinates[:,0].max()]
boundsy=[sorted_coordinates[:,1].min(),sorted_coordinates[:,1].max()]

from scipy.interpolate import CloughTocher2DInterpolator
interpx = LinearNDInterpolator(list(sorted_coordinates), sorted_centers[:,0],)
interpy = LinearNDInterpolator(list(sorted_coordinates), sorted_centers[:,1],)


backcenter=np.array([interpx(new_coordinate),interpy(new_coordinate)]).flatten()


#%% FOM BVEST MANUAL AFFINE

final_gabor_mask_coordinates=reregistered_gabor_mask_full_fov.squeeze()
final_map_mask_coordinates=registered_map_mask_full_fov.squeeze()


firscoord=final_gabor_mask_coordinates[31,:]
interpfristcoord=np.array([interpx(firscoord),interpy(firscoord)]).flatten()


interpolated_gabor_coordinates=np.array([np.array([interpx(new_x_y),interpy(new_x_y)]).flatten() for new_x_y in final_gabor_mask_coordinates])
interpolated_map_coordinates=np.array([np.array([interpx(new_x_y),interpy(new_x_y)]).flatten() for new_x_y in final_map_mask_coordinates])
within_interpolated_gabor_coordinates = interpolated_gabor_coordinates[~np.isnan(interpolated_gabor_coordinates).any(axis=1)]
within_interpolated_map_coordinates = interpolated_map_coordinates[~np.isnan(interpolated_map_coordinates).any(axis=1)]

#%%
f,ax=plt.subplots(1,2)
for contour in external_contours:
    contour = contour.squeeze()
    ax[0].plot(contour[:, 0], contour[:, 1], color='r', linewidth=1)

# Plot contour centers
ax[0].imshow(grid_image,cmap='gray')
ax[0].plot(firscoord[0],firscoord[1],'cx')
ax[0].plot(new_coordinate[0],new_coordinate[1],'cx')
ax[0].scatter(sorted_coordinates[:, 0], sorted_coordinates[:, 1], color='blue', s=10, label="Contour Centers")
ax[0].scatter(final_gabor_mask_coordinates[:, 0].astype('int16'), final_gabor_mask_coordinates[:, 1].astype('int16'),s=1,color='cyan')
ax[0].scatter(final_map_mask_coordinates[:, 0].astype('int16'), final_map_mask_coordinates[:, 1].astype('int16'),s=1,color='cyan')




ax[1].imshow(A, cmap='gray')
ax[1].scatter(square_centers[:, 1], square_centers[:, 0], color='red', s=10, label="Detected Centers")
ax[1].plot(interpfristcoord[0],interpfristcoord[1],'cx')
ax[1].plot(backcenter[0],backcenter[1],'cx')
ax[1].scatter(within_interpolated_gabor_coordinates[:, 0], within_interpolated_gabor_coordinates[:, 1],s=1, color='cyan')
ax[1].scatter(within_interpolated_map_coordinates[:, 0], within_interpolated_map_coordinates[:, 1],s=1, color='cyan')





#%% TESTH THIN PLATE
import skimage as ski

src=sorted_coordinates
dst=sorted_centers
tps = ski.transform.ThinPlateSplineTransform()
tps.estimate(dst, src)
warped = ski.transform.warp(grid_image, tps)
grid_image.shape
A.shape
warped.shape

grid_reference_with_mask = grid_image.copy()
cv2.polylines(grid_reference_with_mask, [np.int32(final_gabor_mask_coordinates)], True, (255, 255, 255), 2, cv2.LINE_AA)
cv2.polylines(grid_reference_with_mask, [np.int32(final_map_mask_coordinates)], True, (255, 255, 255), 2, cv2.LINE_AA)

warped = ski.transform.warp(grid_reference_with_mask, tps)
clipped = warped[:A.shape[0]-1,:grid_image.shape[1]-1]
f,ax=plt.subplots(1,3)
ax[0].imshow(grid_reference_with_mask,cmap='gray')
ax[1].imshow(A,cmap='gray')
ax[2].imshow(clipped,cmap='gray')





