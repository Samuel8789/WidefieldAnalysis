# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:37:51 2025

@author: sp3660
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import pickle
from datetime import datetime
"""
Registration Pipeline.
Every new day of recording the previoue session is registered to the current FOV in the camera.
We take 
Affinee instead of homography

"""

image_folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Animal3p020'


# PATH MANAGING
gabor_reference = cv2.imread(image_folder+r'\WholeBrain_GaborMappingDay.jpg')

gabor_roi = cv2.imread(image_folder+r'\ROI_GaborMappingDay.jpg')
gabor_mask = cv2.imread(image_folder+r'\gabor_mask_Inner.png') # The size of all inner,middle, outer gabor images have to be same.
gabor_mask2 = cv2.imread(image_folder+r'\gabor_mask_Middle.png') # The size of all inner,middle, outer gabor images have to be same.
gabor_mask3 = cv2.imread(image_folder+r'\gabor_mask_Outer.png') # The size of all inner,middle, outer gabor images have to be same.

map_reference = cv2.imread(image_folder+r'\WholeBrain_RetinotopicMappingDay.jpg')

map_roi= cv2.imread(image_folder+r'\ROI_RetinotopicMappingDay.jpg')
map_mask = cv2.imread(image_folder+r'\retinotopy_mask.png')

#%% 



gray_gabor_ref = cv2.cvtColor(gabor_reference, cv2.COLOR_BGR2GRAY)
gray_gabor_roi = cv2.cvtColor(gabor_roi, cv2.COLOR_BGR2GRAY)

gray_gabor_mask1 = cv2.cvtColor(gabor_mask, cv2.COLOR_BGR2GRAY) # The size of all inner,middle, outer gabor images have to be same.
gray_gabor_mask2 = cv2.cvtColor(gabor_mask2, cv2.COLOR_BGR2GRAY) # The size of all inner,middle, outer gabor images have to be same.
gray_gabor_mask3 = cv2.cvtColor(gabor_mask3, cv2.COLOR_BGR2GRAY) # The size of all inner,middle, outer gabor images have to be same.

gray_mapr_ref = cv2.cvtColor(map_reference, cv2.COLOR_BGR2GRAY)
gray_map_roi = cv2.cvtColor(map_roi, cv2.COLOR_BGR2GRAY)
gray_map_mask = cv2.cvtColor(map_mask, cv2.COLOR_BGR2GRAY)

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
def save_affine_transform(affine_matrix, image_folder, prefix="affine_transform"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(image_folder, f"{prefix}_{timestamp}.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(affine_matrix, f)
    print(f"Affine transform saved as: {filename}")

# Load affine transform function
def load_affine_transform(image_folder):
    files = [f for f in os.listdir(image_folder) if f.endswith('.pkl')]
    if not files:
        print("No saved affine transforms found.")
        return None
    
    print("Available transforms:")
    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    
    choice = int(input("Select a transform to load (enter the number): ")) - 1
    if 0 <= choice < len(files):
        with open(os.path.join(image_folder, files[choice]), 'rb') as f:
            affine_matrix = pickle.load(f)
        print(f"Loaded transform from: {files[choice]}")
        return affine_matrix
    else:
        print("Invalid choice.")
        return None

# Visual comparison function
def compare_affine_transforms(image, transforms, corners):
    f, axes = plt.subplots(1, len(transforms), figsize=(15, 5))
    for i, affine_matrix in enumerate(transforms):
        transformed_corners = cv2.transform(corners.reshape(-1, 1, 2), affine_matrix)
        transformed_image = image.copy()
        pts = transformed_corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(transformed_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        axes[i].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Transform {i + 1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    
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
#%%
# After running this cell, I can check affine_matrix (select and F9)
affine_matrix = register_low_mag(
    gray_mapr_ref, gray_gabor_ref, match_to_use=None, eps=10, min_samples=2, 
      angle_deviation_threshold=2, manual=True)

save_affine_transform(affine_matrix, image_folder)
affine_matrix=load_affine_transform(image_folder)
# SAVE TRANSFORM
# LOAD ALL AVAILABLE AND REVIEW WICH ONE IS BETTER


#%% MANUALLY MATHICNG ROI
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def get_roi_coordinates(low_mag_input, roi_input, match_to_use=10, eps=10, min_samples=2, angle_deviation_threshold=2, manual=None):
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
        # SAVE THIS TRANSFORM AND THEN MAKE OPTIONS TO UPLOAD IT



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
    gray_gabor_ref, gray_gabor_roi, match_to_use=None, eps=10, min_samples=2, 
      angle_deviation_threshold=2, manual=True)

map_roi_coordinates = get_roi_coordinates(
    gray_mapr_ref, gray_map_roi, match_to_use=None, eps=10, min_samples=2, 
     angle_deviation_threshold=2, manual=True)

gabor_roi_coordinates_registered = cv2.transform(gabor_roi_coordinates.reshape(-1, 1, 2), affine_matrix)


#%%
loaded_trans_1=load_affine_transform(image_folder)
loaded_trans_2=load_affine_transform(image_folder)

all_transforms = [loaded_trans_1,loaded_trans_2]  # Add all calculated affine matrices here
compare_affine_transforms(gabor_reference, all_transforms, gabor_roi_coordinates)


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

proj_ref = cv2.imread(image_folder+r'\WholeBrain_ExperimentDay.jpg')
# proj_roi= cv2.imread(image_folder+r'\ROI_ExperimentDay.jpg')
# proj_mask = cv2.imread(image_folder+r'\projector_mask.jpg')
gray_proj_ref = cv2.cvtColor(proj_ref, cv2.COLOR_BGR2GRAY)
# gray_proj_roi = cv2.cvtColor(proj_roi, cv2.COLOR_BGR2GRAY)
# gray_proj_mask = cv2.cvtColor(proj_mask, cv2.COLOR_BGR2GRAY)




affine_matrix2 = register_low_mag(
    gray_proj_ref,gray_mapr_ref, match_to_use=None, eps=10, min_samples=2, 
      angle_deviation_threshold=2, manual=True)


# proj_roi_coordinates = get_roi_coordinates(
#     gray_proj_ref, gray_proj_roi, match_to_use=None, eps=10, min_samples=2, 
#      angle_deviation_threshold=2, manual=True)



# proj_mask_full_fov=get_translated_mask_coordinates(proj_roi_coordinates.reshape(-1, 1, 2), gray_proj_mask)





reregistered_gabor_roi_coordinates= cv2.transform(gabor_roi_coordinates_registered.reshape(-1, 1, 2), affine_matrix2)
reregistered_gabor_mask_full_fov= cv2.transform(registered_gabor_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)
reregistered_gabor2_mask_full_fov= cv2.transform(registered_gabor2_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)
reregistered_gabo3r_mask3_full_fov= cv2.transform(registered_gabor3_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)

registered_map_roi_coordinates= cv2.transform(map_roi_coordinates.reshape(-1, 1, 2), affine_matrix2)
registered_map_mask_full_fov= cv2.transform(map_mask_full_fov.astype(np.float32).reshape(-1, 1, 2), affine_matrix2)

#%%
#plot the registration to experiment day


# common_reference2 = map_reference.copy()
# cv2.polylines(common_reference2, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.drawContours(common_reference2, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)

# cv2.polylines(common_reference2, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.drawContours(common_reference2, [registered_gabor_mask_full_fov], -1, (255, 0, 0), 2)
# cv2.drawContours(common_reference2, [registered_gabor2_mask_full_fov], -1, (255, 0, 0), 2)
# cv2.drawContours(common_reference2, [registered_gabor3_mask_full_fov], -1, (255, 0, 0), 2)

# proj_reference_with_mask = proj_ref.copy()
# cv2.polylines(proj_reference_with_mask, [np.int32(proj_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
# for rect in proj_mask_full_fov:
#     cv2.drawContours(proj_reference_with_mask, [np.int32(rect)], -1, (255, 0, 255), 2)


# f,ax=plt.subplots(1,2)
# ax[0].imshow(cv2.cvtColor(common_reference2, cv2.COLOR_BGR2RGB))
# ax[1].imshow(cv2.cvtColor(proj_reference_with_mask, cv2.COLOR_BGR2RGB))

# #%%

# final_reference = proj_ref.copy()
# cv2.polylines(final_reference, [np.int32(registered_map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.drawContours(final_reference, [np.int32(registered_map_mask_full_fov)], -1, (255, 0, 255), 2)
# cv2.polylines(final_reference, [np.int32(reregistered_gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.drawContours(final_reference, [np.int32(reregistered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)

# cv2.polylines(final_reference, [np.int32(proj_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
# for rect in proj_mask_full_fov:
#     cv2.drawContours(final_reference, [np.int32(rect)], -1, (255, 0, 255), 2)


# f,ax=plt.subplots(1)
# ax.imshow(cv2.cvtColor(final_reference, cv2.COLOR_BGR2RGB))



#%% CREATE COMBINED MASK

final_reference = proj_ref.copy()
cv2.drawContours(final_reference, [np.int32(registered_map_mask_full_fov)], -1, (255, 0, 255), 2)
cv2.drawContours(final_reference, [np.int32(reregistered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)
f,ax=plt.subplots(1)
ax.imshow(cv2.cvtColor(final_reference, cv2.COLOR_BGR2RGB))






#%%


# # THIS CONVERTS GABOR AND V1 MASK TO PROJECOT RESPACE
# new_roi_coords_dest_gabor = reregistered_gabor_mask_full_fov.astype('float32').reshape(-1, 1, 2)
# new_roi_coords_dest=registered_map_mask_full_fov.astype('float32').reshape(-1, 1, 2)


# # Apply the inverse homography to the new ROI coordinates
# new_roi_coords_src = cv2.perspectiveTransform(new_roi_coords_dest, H_inv)
# new_roi_coords_src_gabor = cv2.perspectiveTransform(new_roi_coords_dest_gabor, H_inv)
#%%
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
#
# Save the mask as a JPEG file (optional)
cv2.imwrite(r'C:\Users\calci\Desktop\SamAnalysisTemp\FullRegistration\binary_mask_v1.jpg', binary_mask)

binary_mask_gabor = np.zeros((height, width), dtype=np.uint8)

# Define the coordinates of the polygon (for example purposes)

polygon_coords_gabor=new_roi_coords_src_gabor.squeeze().astype('int32')

# Draw the polygon on the image
cv2.polylines(binary_mask_gabor, [polygon_coords_gabor], isClosed=True, color=255, thickness=2)

# Fill the polygon (if needed, to make the mask solid)
cv2.fillPoly(binary_mask_gabor, [polygon_coords_gabor], color=255)


# Save the mask as a JPEG file (optional)
cv2.imwrite(r'C:\Users\calci\Desktop\SamAnalysisTemp\FullRegistration\binary_mask_gabor.jpg', binary_mask_gabor)