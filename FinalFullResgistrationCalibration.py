# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:49:56 2025

@author: sp3660
Final script to register cortex across experimental days, overlapping gabor and retinotopic mas, 
performing grid calibration of the projector and registering active areas on the final experimental session
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import pickle
from datetime import datetime
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
import skimage as ski
from pathlib import Path
"""
Registration Pipeline.
Every new day of recording the previoue session is registered to the current FOV in the camera.
We take 
Affinee instead of homography

"""
mouse=r'3p001'
image_folder=rf'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Animal{mouse}'
output_dir = rf'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Animal{mouse}'
# PATH MANAGING
gabor_reference = cv2.imread(image_folder+r'\WholeBrain_GaborMappingDay.jpg')
gabor_roi = cv2.imread(image_folder+r'\ROI_GaborMappingDay.jpg')
gabor_mask = cv2.imread(image_folder+r'\gabor_mask_Inner.png') # The size of all inner,middle, outer gabor images have to be same.
gabor_mask2 = cv2.imread(image_folder+r'\gabor_mask_Middle.png') # The size of all inner,middle, outer gabor images have to be same.
gabor_mask3 = cv2.imread(image_folder+r'\gabor_mask_Outer.png') # The size of all inner,middle, outer gabor images have to be same.
map_reference = cv2.imread(image_folder+r'\WholeBrain_RetinotopicMappingDay.jpg')
map_roi= cv2.imread(image_folder+r'\ROI_RetinotopicMappingDay.jpg')
map_mask = cv2.imread(image_folder+r'\retinotopy_mask.png')

proj_ref = cv2.imread(image_folder+r'\WholeBrain_ExperimentDay.jpg')
# PATH MANAGING
gabor_reference = cv2.imread(image_folder+r'\WholeBrain_GaborMappingDay.jpg')
grid_file = image_folder+r'\grid_2025130.mat'
# Load the .tif image (distorted projection of the grid)
grid_image_file = image_folder+r'\Grid_snapshot.jpg'


remove_last_row=False

#%% 
gray_mapr_ref=np.empty((0))
gray_proj_ref=np.empty((0))
affine_matrix=np.empty((0))
affine_matrix2=np.empty((0))

gray_gabor_ref = cv2.cvtColor(gabor_reference, cv2.COLOR_BGR2GRAY)
gray_gabor_roi = cv2.cvtColor(gabor_roi, cv2.COLOR_BGR2GRAY)
gray_gabor_mask1 = cv2.cvtColor(gabor_mask, cv2.COLOR_BGR2GRAY) # The size of all inner,middle, outer gabor images have to be same.
gray_gabor_mask2 = cv2.cvtColor(gabor_mask2, cv2.COLOR_BGR2GRAY) # The size of all inner,middle, outer gabor images have to be same.
gray_gabor_mask3 = cv2.cvtColor(gabor_mask3, cv2.COLOR_BGR2GRAY) # The size of all inner,middle, outer gabor images have to be same.
gray_mapr_ref = cv2.cvtColor(map_reference, cv2.COLOR_BGR2GRAY)
gray_map_roi = cv2.cvtColor(map_roi, cv2.COLOR_BGR2GRAY)
gray_map_mask = cv2.cvtColor(map_mask, cv2.COLOR_BGR2GRAY)


gray_proj_ref = cv2.cvtColor(proj_ref, cv2.COLOR_BGR2GRAY)
data = loadmat(grid_file)
A = data['fullgrid']  # Extract the array from the .mat file
grid_image = io.imread(grid_image_file)

plt.close('all')
f,ax=plt.subplots(3,2)
f.tight_layout()
ax.flatten()[0].imshow(gray_gabor_ref)
ax.flatten()[1].imshow(gray_mapr_ref)
ax.flatten()[2].imshow(gray_gabor_roi)
ax.flatten()[3].imshow(gray_map_roi)
ax.flatten()[4].imshow(gray_gabor_mask1+gray_gabor_mask2+gray_gabor_mask3)
ax.flatten()[5].imshow(gray_map_mask)

#%% NEW SAVING FUCNTIONS 


# Save affine transform function
def save_affine_transform(affine_matrix, image_folder, source, target):
    """
    Save an affine transformation matrix with metadata about the registered images.
    
    Parameters:
    affine_matrix (np.array): The affine transformation matrix.
    image_folder (str): The directory where the file should be saved.
    source (str): The name of the source image.
    target (str): The name of the target image.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(image_folder) / f"affine_{source}_to_{target}_{timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(affine_matrix, f)
    print(f"Affine transform saved as: {filename}")

# Load affine transform function
def load_affine_transform(image_folder,first,second):
    files = [f for f in os.listdir(image_folder) if f.endswith('.pkl') and (first in f) and (second in f)]
    if not files:
        print("No saved affine transforms found.")
        return np.empty((0))
    
    print("Available transforms:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    
    choice = int(input(f"Select a transform to load for {first} and {second} registration (enter the number): ")) - 1
    if 0 <= choice < len(files):
        with open(os.path.join(image_folder, files[choice]), 'rb') as f:
            affine_matrix = pickle.load(f)
        print(f"Loaded transform from: {files[choice]}")
        return affine_matrix
    else:
        print("Invalid choice.")
        return np.empty((0))

    
#%% MANUALLY MATHIOCNG LOW MAG IMAGES
def register_low_mag(low_mag_input, roi_input, match_to_use=10, eps=10, min_samples=2, angle_deviation_threshold=2, manual=None):
    global low_mag, roi
    low_mag = low_mag_input.copy()
    roi = roi_input.copy()
    # Initialize SIFT detector with parameters
   
    global manual_matches1, manual_matches2
    # Manual matching variables
    manual_matches1 = []  # Points from img1
    manual_matches2 = []  # Points from img2
    
    h, w = roi.shape[:2]
    corners_smaller = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype='float32').reshape(-1, 1, 2)


    def select_points(event, x, y, flags, param):
        """Handle manual point selection."""
        global manual_matches1, manual_matches2, low_mag, roi
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if param == 'image1':
                manual_matches1.append((x, y))
                cv2.circle(low_mag, (x, y), 2, (255, 255, 0), -1)
                cv2.imshow('Low Magnification Image', low_mag)
            elif param == 'image2':
                manual_matches2.append((x, y))
                cv2.circle(roi, (x, y), 2, (255, 255, 0), -1)
                cv2.imshow('ROI Image', roi)

    # Step 1: Perform matching based on the `match_to_use` parameter
    if manual:  # Manual matching mode
        # Show both images and allow manual point selection
        cv2.imshow('Low Magnification Image', low_mag)
        cv2.setMouseCallback('Low Magnification Image', select_points, param='image1')

        cv2.imshow('ROI Image', roi)
        cv2.setMouseCallback('ROI Image', select_points, param='image2')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Now, 'manual_matches1' and 'manual_matches2' contain the selected points
        if len(manual_matches1) < 2 or len(manual_matches2) < 2:
            raise ValueError("Not enough points selected for manual matching.")
        
        # Manually selected points for homography calculation
        src_pts = np.float32(manual_matches1).reshape(-1, 1, 2)
        dst_pts = np.float32(manual_matches2).reshape(-1, 1, 2)
        affine_matrix, inliers = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC)
        corners_larger=cv2.transform(corners_smaller.reshape(-1, 1, 2), affine_matrix)


    return affine_matrix


#%% IF THERE IS AN AFFINR LOAD IT CHECK IT AND THEN GIVE THE OPTION TO Clculate againb. IF THERE ISENT CALCULATE


affine_matrix=load_affine_transform(image_folder,'map','gabor')
if affine_matrix.any():
    choice = int(input(f"Do you want to redo the trnasform for map and gabor (1 for yes 0 for no): ")) 
elif not affine_matrix.any() or choice:
    #% CALCULATE THE AFFINE AND SAAVE IT
    # After running this cell, I can check affine_matrix (select and F9)
    affine_matrix = register_low_mag(
        gray_mapr_ref, gray_gabor_ref, match_to_use=None, eps=10, min_samples=2, 
          angle_deviation_threshold=2, manual=True)
    save_affine_transform(affine_matrix, image_folder,'map','gabor')

if gray_proj_ref.any() and gray_mapr_ref.any() :
    affine_matrix2=load_affine_transform(image_folder,'exp','map')
    if affine_matrix2.any():
        choice = int(input(f"Do you want to redo the trnasform for exp and map (1 for yes 0 for no): ")) 
    elif not affine_matrix2.any() or choice:
      affine_matrix2 = register_low_mag(
            gray_proj_ref,gray_mapr_ref, match_to_use=None, eps=10, min_samples=2, 
              angle_deviation_threshold=2, manual=True)
      save_affine_transform(affine_matrix2, image_folder,'exp','map')

    
    
    #%% MANUALLY MATHICNG ROIs to WHOLE BRAIN IUMAGE
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    def get_roi_coordinates(low_mag_input, roi_input,roi_name,whole_name, match_to_use=10, eps=10, min_samples=2, angle_deviation_threshold=2, manual=None):
        global low_mag, roi
        low_mag = low_mag_input.copy()
        roi = roi_input.copy()
        # Initialize SIFT detector with parameters
       
        global manual_matches1, manual_matches2
        # Manual matching variables
        manual_matches1 = []  # Points from img1
        manual_matches2 = []  # Points from img2
        
        h, w = roi.shape[:2]
        corners_smaller = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype='float32').reshape(-1, 1, 2)


        def select_points(event, x, y, flags, param):
            """Handle manual point selection."""
            global manual_matches1, manual_matches2, low_mag, roi
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if param == 'image1':
                    manual_matches1.append((x, y))
                    cv2.circle(low_mag, (x, y), 2, (255, 255, 0), -1)
                    cv2.imshow('Low Magnification Image', low_mag)
                elif param == 'image2':
                    manual_matches2.append((x, y))
                    cv2.circle(roi, (x, y), 2, (255, 255, 0), -1)
                    cv2.imshow('ROI Image', roi)

        # Step 1: Perform matching based on the `match_to_use` parameter
        if manual:  # Manual matching mode
            affine_matrix=np.empty((0))
            affine_matrix=load_affine_transform(image_folder,roi_name,whole_name)
            if affine_matrix.any():
                choice = int(input(f"Do you want to redo the trnasform for {roi_name} and {whole_name} (1 for yes 0 for no): ")) 

            elif not affine_matrix.any() or choice:
                # Show both images and allow manual point selection
                cv2.imshow('Low Magnification Image', low_mag)
                cv2.setMouseCallback('Low Magnification Image', select_points, param='image1')
    
                cv2.imshow('ROI Image', roi)
                cv2.setMouseCallback('ROI Image', select_points, param='image2')
    
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
                # Now, 'manual_matches1' and 'manual_matches2' contain the selected points
                if len(manual_matches1) < 2 or len(manual_matches2) < 2:
                    raise ValueError("Not enough points selected for manual matching.")
                
                # Manually selected points for homography calculation
                src_pts = np.float32(manual_matches1).reshape(-1, 1, 2)
                dst_pts = np.float32(manual_matches2).reshape(-1, 1, 2)
                affine_matrix, inliers = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC)
                # SAVE THIS TRANSFORM AND THEN MAKE OPTIONS TO UPLOAD IT
                save_affine_transform(affine_matrix, image_folder,roi_name,whole_name)
            corners_larger=cv2.transform(corners_smaller.reshape(-1, 1, 2), affine_matrix)





        else:  # Automatic SIFT matching mode
            # Initialize SIFT detector with parameters
            sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10)
     
            # Detect keypoints and descriptors for both images
            kp_larger, des_larger = sift.detectAndCompute(low_mag, None)
            kp_smaller, des_smaller = sift.detectAndCompute(roi, None)
     
            # Create BFMatcher object (using KNN instead of direct matching)
            bf = cv2.BFMatcher(cv2.NORM_L2)
     
            # Match descriptors using KNN
            matches = bf.knnMatch(des_smaller, des_larger, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 1 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)
     
            # Sort matches based on distance (after ratio test)
            good_matches = sorted(good_matches, key=lambda x: x.distance)
     
            # Check if enough good matches are found
            if len(good_matches) < 5:
                raise ValueError("Not enough matches to compute homography.")
     
            # Extract location of good matches (top matches)
            top_matches = good_matches[:match_to_use]
            img_matches = cv2.drawMatches(roi, kp_smaller, low_mag, kp_larger, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            f, ax = plt.subplots()
            ax.imshow(img_matches)
            f.suptitle(f'Top {len(matches)} Matches After Clustering, Length and Angle Filtering')
            plt.show()
     
            # Extract positions and compute angles and lengths
            positions_and_angles = []
            angles = []
            lengths = []
            for m in top_matches:
                # Get the positions of the matched keypoints
                pt_smaller = kp_smaller[m.queryIdx].pt  # Point in the ROI (smaller image)
                pt_larger = kp_larger[m.trainIdx].pt   # Point in the larger image
     
                # Calculate the angle between the points
                dx = pt_larger[0] - pt_smaller[0]
                dy = pt_larger[1] - pt_smaller[1]
                angle = np.degrees(np.arctan2(dy, dx))  # Angle in degrees
     
                # Calculate the length of the line (distance between the two points)
                length = np.sqrt(dx**2 + dy**2)
     
                # Store the positions, angle, and length
                positions_and_angles.append((pt_smaller, pt_larger, angle, length))
                angles.append(angle)
                lengths.append(length)
     
            # Reshape angles for clustering
            angles = np.array(angles).reshape(-1, 1)
     
            # Apply DBSCAN clustering to group lines with similar angles
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(angles)
     
            # Find the largest cluster (most frequent label)
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster_label = unique_labels[np.argmax(counts)]
     
            # Find the average angle of the largest cluster (mean angle)
            largest_cluster_angles = [angles[i] for i in range(len(angles)) if labels[i] == largest_cluster_label]
            mean_angle = np.mean(largest_cluster_angles)
     
            # Filter matches that belong to the largest cluster and are within the angular deviation threshold
            filtered_matches = []
            filtered_positions_and_angles = []
            for i, label in enumerate(labels):
                if label == largest_cluster_label:
                    # Calculate the angular deviation
                    angle_deviation = abs(angles[i] - mean_angle)
                    if angle_deviation <= angle_deviation_threshold:
                        # Filter based on line length
                        filtered_matches.append(top_matches[i])
                        filtered_positions_and_angles.append(positions_and_angles[i])
     
            # Draw the matches on the image
            img_matches = cv2.drawMatches(roi, kp_smaller, low_mag, kp_larger, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            f, ax = plt.subplots()
            ax.imshow(img_matches)
            f.suptitle(f'Top {len(filtered_matches)} Matches After Clustering, Length and Angle Filtering')
            plt.show()
     
            # Extract points for homography
            src_pts = np.float32([kp_smaller[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_larger[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            corners_larger = cv2.perspectiveTransform(corners_smaller, H)




        corners_larger = corners_larger.reshape(4, 1, 2)

        # Draw the polygon (corners connected) on the larger image
        low_mag_with_polygon = low_mag.copy()
        corners_larger = corners_larger.reshape(-1, 2)  # Flatten to 2D array for drawing
        pts = corners_larger.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw the polygon with a specific color and thickness
        cv2.polylines(low_mag_with_polygon, [pts], isClosed=True, color=(255, 255, 0), thickness=3)

        # Plot the polygon on the larger image
        plt.figure(figsize=(8, 8))
        plt.imshow(low_mag_with_polygon, cmap='gray')
        plt.title('Mapped ROI Polygon on Larger Image')
        plt.show()

        return corners_larger

    gabor_roi_coordinates = get_roi_coordinates(
        gray_gabor_ref, gray_gabor_roi,'gabor_roi','gabor_whole', match_to_use=None, eps=10, min_samples=2, 
          angle_deviation_threshold=2, manual=True)

    map_roi_coordinates = get_roi_coordinates(
        gray_mapr_ref, gray_map_roi,'map_roi','map_whole', match_to_use=None, eps=10, min_samples=2, 
         angle_deviation_threshold=2, manual=True)

    gabor_roi_coordinates_registered = cv2.transform(gabor_roi_coordinates.reshape(-1, 1, 2), affine_matrix)
    
#%% GET MASK COORDINATES ON BIG TARGET IMAGE

def get_translated_mask_coordinates(roi_coordinates, mask):

    # Define ROI coordinates in the reference image (top-left corner of the mask in the reference image)
    roi_x, roi_y =np.int32(roi_coordinates)[0][0]
     # Replace with actual ROI coordinates
    
    # Find contours of the mask
    _, binary_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Assuming the mask has one contour (outline)
    
    if len(contours)==1:
       mask_contour = contours[0]
       mask_contour_translated = mask_contour + np.array([roi_x, roi_y])

    else:

        # # Create a new mask to draw all contours
        mask_contour = np.zeros_like(mask)        
        # Draw each contour on the new mask
        mask_contour_translated=[]
        for contour in contours:
            mask_contour_translated.append( contour + np.array([roi_x, roi_y]))


    return mask_contour_translated

gabor_mask_full_fov=get_translated_mask_coordinates(gabor_roi_coordinates.reshape(-1, 1, 2), gray_gabor_mask1)
gabor2_mask_full_fov=get_translated_mask_coordinates(gabor_roi_coordinates.reshape(-1, 1, 2), gray_gabor_mask2)
gabor3_mask_full_fov=get_translated_mask_coordinates(gabor_roi_coordinates.reshape(-1, 1, 2), gray_gabor_mask3)


registered_gabor_mask_full_fov=cv2.transform(gabor_mask_full_fov.reshape(-1, 1, 2), affine_matrix)
registered_gabor2_mask_full_fov=cv2.transform(gabor2_mask_full_fov.reshape(-1, 1, 2), affine_matrix)
registered_gabor3_mask_full_fov=cv2.transform(gabor2_mask_full_fov.reshape(-1, 1, 2), affine_matrix)


map_mask_full_fov=get_translated_mask_coordinates(map_roi_coordinates.reshape(-1, 1, 2), gray_map_mask)
#%%
#plot the original images with roi and masks without trnasfrom side by side


gabor_reference_with_roi = gabor_reference.copy()
cv2.polylines(gabor_reference_with_roi, [np.int32(gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(gabor_reference_with_roi, [gabor_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(gabor_reference_with_roi, [gabor2_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(gabor_reference_with_roi, [gabor3_mask_full_fov], -1, (255, 0, 0), 2)

map_reference_with_mask = map_reference.copy()
cv2.polylines(map_reference_with_mask, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(map_reference_with_mask, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)

f,ax=plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(map_reference_with_mask, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(gabor_reference_with_roi, cv2.COLOR_BGR2RGB))

#plot the original images with roi and masks without trnasfrom side by side
height, width = map_reference.shape[:2]
aligned_gabor = cv2.warpAffine(gabor_reference, affine_matrix, (width, height))

gabor_reference_with_roi = aligned_gabor.copy()
cv2.polylines(gabor_reference_with_roi, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(gabor_reference_with_roi, [registered_gabor_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(gabor_reference_with_roi, [registered_gabor2_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(gabor_reference_with_roi, [registered_gabor3_mask_full_fov], -1, (255, 0, 0), 2)

map_reference_with_mask = map_reference.copy()
cv2.polylines(map_reference_with_mask, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(map_reference_with_mask, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)

f,ax=plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(map_reference_with_mask, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(gabor_reference_with_roi, cv2.COLOR_BGR2RGB))
refs=[[661,554], [576,598]]
for ref in refs:
    ax[0].plot(ref[0], ref[1], 'wo', markersize=2)  # White point on the first subplot
    ax[1].plot(ref[0], ref[1], 'wo', markersize=2)  # White point on the second subplot



common_reference = map_reference.copy()
cv2.polylines(common_reference, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(common_reference, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)

cv2.polylines(common_reference, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(common_reference, [registered_gabor_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(common_reference, [registered_gabor2_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(common_reference, [registered_gabor3_mask_full_fov], -1, (255, 0, 0), 2)



f,ax=plt.subplots(1,2)
ax[0].imshow(cv2.cvtColor(common_reference, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(common_reference, cv2.COLOR_BGR2RGB))


#%% PROJECTOR DAY (Experimental day)

# REGISTERING THE REFERENCE FROM LAST DAY TO THE NEW PROJECTRO REFREENCE 


reregistered_gabor_roi_coordinates= cv2.transform(gabor_roi_coordinates_registered.reshape(-1, 1, 2), affine_matrix2)
reregistered_gabor_mask_full_fov= cv2.transform(registered_gabor_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)
reregistered_gabor2_mask_full_fov= cv2.transform(registered_gabor2_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)
reregistered_gabor3_mask_full_fov= cv2.transform(registered_gabor3_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)

registered_map_roi_coordinates= cv2.transform(map_roi_coordinates.reshape(-1, 1, 2), affine_matrix2)
registered_map_mask_full_fov= cv2.transform(map_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)

#%%
common_reference = map_reference.copy()
cv2.polylines(common_reference, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(common_reference, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)

cv2.polylines(common_reference, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(common_reference, [registered_gabor_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(common_reference, [registered_gabor2_mask_full_fov], -1, (255, 0, 0), 2)
cv2.drawContours(common_reference, [registered_gabor3_mask_full_fov], -1, (255, 0, 0), 2)




final_reference = proj_ref.copy()
cv2.polylines(final_reference, [np.int32(registered_map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(final_reference, [np.int32(registered_map_mask_full_fov)], -1, (255, 0, 255), 2)
cv2.polylines(final_reference, [np.int32(reregistered_gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(final_reference, [np.int32(reregistered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)
cv2.drawContours(final_reference, [np.int32(reregistered_gabor2_mask_full_fov)], -1, (255, 0, 0), 2)
cv2.drawContours(final_reference, [np.int32(reregistered_gabor3_mask_full_fov)], -1, (255, 0, 0), 2)
f,ax=plt.subplots(1,2)
ax[0].imshow(cv2.cvtColor(common_reference, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(final_reference, cv2.COLOR_BGR2RGB))




#%%


# Step 1: Detect edges in the .tif image using Canny edge detector
edges = feature.canny(grid_image, sigma=2)

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
size_threshold = 50
filtered_contours = []
# Step 5: Calculate the center of each external contour
contour_centers = []
for contour in external_contours:
    contour = contour.squeeze()  # Remove unnecessary dimensions for processing
    area=cv2.contourArea(contour)
    print(area)
    if area > size_threshold:
        M = cv2.moments(contour)
        if M['m00'] != 0:  # Avoid division by zero
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            contour_centers.append([cX, cY])
            filtered_contours.append(contour)
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

if remove_last_row:
    max_y = np.max(square_centers[:, 0])  # Find the maximum y-value
    filtered_centers = square_centers[square_centers[:, 0] < max_y]  # Remove points with max y
 
else:
    filtered_centers=square_centers
    
    
sorted_indices = np.lexsort((filtered_centers[:, 0], filtered_centers[:, 1]))  # Sort by column 0, then column 1
sorted_centers = filtered_centers[sorted_indices]
sorted_centers[:, [0, 1]] = sorted_centers[:, [1, 0]]
#%%
f,ax=plt.subplots(1,2)
ax[0].imshow(grid_image, cmap='gray')
for contour in filtered_contours:
    contour = contour.squeeze()
    ax[0].plot(contour[:, 0], contour[:, 1], color='r', linewidth=1)

# Plot contour centers
ax[0].scatter(sorted_coordinates[:, 0], sorted_coordinates[:, 1], color='blue', s=10, label="Contour Centers")
ax[1].imshow(A, cmap='gray')
ax[1].scatter(square_centers[:, 1], square_centers[:, 0], color='red', s=10, label="Detected Centers")
#%% TESTH THIN PLATE

final_gabor_mask_coordinates=reregistered_gabor_mask_full_fov.squeeze()
final_map_mask_coordinates=registered_map_mask_full_fov.squeeze()
final_gabor2_mask_coordinates=reregistered_gabor2_mask_full_fov.squeeze()
final_gabor3_mask_coordinates=reregistered_gabor3_mask_full_fov.squeeze()

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

#%% FINAL MASK CREATION 
# Function to map FOV to projector
def map_fov_to_projector(fov_image, tps, grid):
    warped = ski.transform.warp(fov_image, tps)
    clipped = warped[:grid.shape[0] , :fov_image.shape[1] ]
    # Convert to uint8 format (0-255)
    clipped_uint8 = (clipped * 255).astype(np.uint8)
    return clipped_uint8

# Assuming final_gabor_mask_coordinates and final_map_mask_coordinates are defined
height, width = grid_image.shape

# Create blank (black) images
binary_mask_map = np.zeros((height, width), dtype=np.uint8)
binary_mask_gabor = np.zeros((height, width), dtype=np.uint8)
binary_mask_gabor2 = np.zeros((height, width), dtype=np.uint8)
binary_mask_gabor3 = np.zeros((height, width), dtype=np.uint8)

# Fill the polygons
cv2.fillPoly(binary_mask_map, [np.int32(final_map_mask_coordinates)], color=(255, 255, 255))
cv2.fillPoly(binary_mask_gabor, [np.int32(final_gabor_mask_coordinates)], color=(255, 255, 255))
cv2.fillPoly(binary_mask_gabor2, [np.int32(final_gabor2_mask_coordinates)], color=(255, 255, 255))
cv2.fillPoly(binary_mask_gabor3, [np.int32(final_gabor3_mask_coordinates)], color=(255, 255, 255))

# Subtract the overlapping regions
final_region1 = cv2.subtract(binary_mask_map, binary_mask_gabor)
final_region2 = cv2.subtract(binary_mask_map, binary_mask_gabor2)
final_region3 = cv2.subtract(binary_mask_map, binary_mask_gabor3)

# Apply transformations
final_mask_gabor1 = map_fov_to_projector(binary_mask_gabor, tps, A)
final_mask_gabor2 = map_fov_to_projector(binary_mask_gabor2, tps, A)
final_mask_gabor3 = map_fov_to_projector(binary_mask_gabor3, tps, A)
final_mask_fullv1 = map_fov_to_projector(binary_mask_map, tps, A)
final_mask_v1_nogabor1 = map_fov_to_projector(final_region1, tps, A)
final_mask_v1_nogabor2 = map_fov_to_projector(final_region2, tps, A)
final_mask_v1_nogabor3 = map_fov_to_projector(final_region3, tps, A)

# Display the masks
f, ax = plt.subplots(2, 4, figsize=(15, 10))
ax[0, 0].imshow(final_mask_gabor1, cmap='gray')
ax[0, 0].set_title('Gabor 1')
ax[0, 1].imshow(final_mask_gabor2, cmap='gray')
ax[0, 1].set_title('Gabor 2')
ax[0, 2].imshow(final_mask_gabor3, cmap='gray')
ax[0, 2].set_title('Gabor 3')
ax[0, 3].imshow(final_mask_fullv1, cmap='gray')
ax[0, 3].set_title('Full V1')
ax[1, 0].imshow(final_mask_v1_nogabor1, cmap='gray')
ax[1, 0].set_title('V1 - Gabor 1')
ax[1, 1].imshow(final_mask_v1_nogabor2, cmap='gray')
ax[1, 1].set_title('V1 - Gabor 2')
ax[1, 2].imshow(final_mask_v1_nogabor3, cmap='gray')
ax[1, 2].set_title('V1 - Gabor 3')

# Hide unused subplots
for i in range(3, 4):
    ax[1, i].axis('off')

plt.tight_layout()
plt.show()

# Save the masks as JPEG files


cv2.imwrite(os.path.join(output_dir, 'final_mask_gabor1.jpg'), final_mask_gabor1)
cv2.imwrite(os.path.join(output_dir, 'final_mask_gabor2.jpg'), final_mask_gabor2)
cv2.imwrite(os.path.join(output_dir, 'final_mask_gabor3.jpg'), final_mask_gabor3)
cv2.imwrite(os.path.join(output_dir, 'final_mask_fullv1.jpg'), final_mask_fullv1)
cv2.imwrite(os.path.join(output_dir, 'final_mask_v1_no_gabor1.jpg'), final_mask_v1_nogabor1)
cv2.imwrite(os.path.join(output_dir, 'final_mask_v1_no_gabor2.jpg'), final_mask_v1_nogabor2)
cv2.imwrite(os.path.join(output_dir, 'final_mask_v1_no_gabor3.jpg'), final_mask_v1_nogabor3)

print(f"Masks saved to directory: {output_dir}")

