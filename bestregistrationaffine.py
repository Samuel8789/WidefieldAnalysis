import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
"""
Registration Pipeline.
Every new day of recording the previoue session is registered to the current FOV in the camera.
We take 
Affinee oimsetad of homography


"""

# image_folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Full3GbaorMap'
# image_folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20241204Issues'
# image_folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20241204Issues2'
image_folder=r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20241225'


# PATH MANAGING
gabor_reference_small = cv2.imread(image_folder+r'\RecBin_NoMag_Gabor.jpg')
gabor_reference = cv2.imread(image_folder+r'\RecBin_NoMag_GaborFull.jpg')

gabor_roi = cv2.imread(image_folder+r'\RecBin_Roi_Gabor.jpg')
gabor_mask = cv2.imread(image_folder+r'\RecBin_Roi_Mask_Gabor_Inner.jpg')
gabor_mask2 = cv2.imread(image_folder+r'\RecBin_Roi_Mask_Gabor_Middle.jpg')
gabor_mask3 = cv2.imread(image_folder+r'\RecBin_Roi_Mask_Gabor_Outer.jpg')



map_reference_small = cv2.imread(image_folder+r'\RecBin_NoMag_Map.jpg')
map_reference = cv2.imread(image_folder+r'\RecBin_NoMag_MapFull.jpg')

map_roi= cv2.imread(image_folder+r'\RecBin_Roi_Map.jpg')
map_mask = cv2.imread(image_folder+r'\RecBin_Roi_Mask_Map.jpg')

if os.path.isfile(image_folder+r'\RecBin_NoMag_Proj.jpg'):
    proj_img = cv2.imread(image_folder+r'\RecBin_NoMag_Proj.jpg')
    proj_roi= cv2.imread(image_folder+r'\RecBin_Roi_Proj.jpg')
    proj_mask = cv2.imread(image_folder+r'\RecBin_Roi_Mask_Proj.jpg')



gray_gabor_ref = cv2.cvtColor(gabor_reference, cv2.COLOR_BGR2GRAY)
gray_gabor_roi = cv2.cvtColor(gabor_roi, cv2.COLOR_BGR2GRAY)
gray_gabor_mask1 = cv2.cvtColor(gabor_mask, cv2.COLOR_BGR2GRAY)
gray_gabor_mask2 = cv2.cvtColor(gabor_mask2, cv2.COLOR_BGR2GRAY)
gray_gabor_mask3 = cv2.cvtColor(gabor_mask3, cv2.COLOR_BGR2GRAY)

gray_mapr_ref = cv2.cvtColor(map_reference, cv2.COLOR_BGR2GRAY)
gray_map_roi = cv2.cvtColor(map_roi, cv2.COLOR_BGR2GRAY)
gray_map_mask = cv2.cvtColor(map_mask, cv2.COLOR_BGR2GRAY)

if os.path.isfile(image_folder+r'\RecBin_NoMag_Proj.jpg'):
    gray_proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2GRAY)
    gray_proj_roi = cv2.cvtColor(proj_roi, cv2.COLOR_BGR2GRAY)
    gray_proj_mask = cv2.cvtColor(proj_mask, cv2.COLOR_BGR2GRAY)

plt.close('all')

f,ax=plt.subplots(3,2)
f.tight_layout()
ax.flatten()[0].imshow(gray_gabor_ref)
ax.flatten()[1].imshow(gray_mapr_ref)
ax.flatten()[2].imshow(gray_gabor_roi)
ax.flatten()[3].imshow(gray_map_roi)
ax.flatten()[4].imshow(gray_gabor_mask1+gray_gabor_mask2+gray_gabor_mask3)
ax.flatten()[5].imshow(gray_map_mask)


 #%% CURATED MATCHING LOW MAG IMAGES
# # Global variables
# matches = []
# accepted_matches = []
# rejected_matches = []
# keypoints_ref = []
# keypoints_target = []
# gray_gabor_ref = None
# gray_mapr_ref = None
# img_matches = None
# affine_matrix = None  # Store the affine matrix globally

# # Function to handle mouse click event
# def select_match(event, x, y, flags, param):
#     global accepted_matches, rejected_matches, img_matches

#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Check if the click was inside the accepted figure (left)
#         if x < img_matches.shape[1] // 2:
#             # Determine which match was clicked
#             for i, match in enumerate(accepted_matches):
#                 kp1 = keypoints_ref[match.queryIdx]
#                 kp2 = keypoints_target[match.trainIdx]
#                 # Check if the click is near the keypoint
#                 if abs(kp1.pt[0] - x) < 10 and abs(kp1.pt[1] - y) < 10:
#                     # Move match to rejected
#                     rejected_matches.append(match)
#                     accepted_matches.remove(match)
#                     update_match_display()
#                     break
#         # Check if the click was inside the rejected figure (right)
#         else:
#             # Determine which match was clicked
#             for i, match in enumerate(rejected_matches):
#                 kp1 = keypoints_ref[match.queryIdx]
#                 kp2 = keypoints_target[match.trainIdx]
#                 # Check if the click is near the keypoint
#                 if abs(kp1.pt[0] - x) < 10 and abs(kp1.pt[1] - y) < 10:
#                     # Move match to accepted
#                     accepted_matches.append(match)
#                     rejected_matches.remove(match)
#                     update_match_display()

# # Update match display for accepted and rejected matches
# def update_match_display():
#     global img_matches
#     img_matches = np.zeros_like(gray_mapr_ref)

#     # Draw accepted matches
#     img_matches = cv2.drawMatches(
#         gray_mapr_ref, keypoints_ref,  # First image and keypoints
#         gray_gabor_ref, keypoints_target,  # Second image and keypoints
#         accepted_matches,  # Accepted matches
#         img_matches,  # The image where matches are drawn
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#     # Create second figure with rejected matches
#     img_rejected_matches = np.zeros_like(gray_mapr_ref)
#     img_rejected_matches = cv2.drawMatches(
#         gray_mapr_ref, keypoints_ref,  # First image and keypoints
#         gray_gabor_ref, keypoints_target,  # Second image and keypoints
#         rejected_matches,  # Rejected matches
#         img_rejected_matches,  # The image where matches are drawn
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#     # Display the updated figures
#     cv2.imshow("Accepted Matches", img_matches)
#     cv2.imshow("Rejected Matches", img_rejected_matches)

# # Function to calculate affine transformation based on accepted matches
# def calculate_affine():
#     global accepted_matches

#     # Extract points from accepted matches
#     points_ref = np.zeros((len(accepted_matches), 2), dtype=np.float32)
#     points_target = np.zeros((len(accepted_matches), 2), dtype=np.float32)
#     for i, match in enumerate(accepted_matches):
#         points_ref[i, :] = keypoints_ref[match.queryIdx].pt
#         points_target[i, :] = keypoints_target[match.trainIdx].pt

#     # Compute the affine transformation matrix
#     affine_matrix, inliers = cv2.estimateAffine2D(points_target, points_ref, method=cv2.RANSAC)

#     # If the affine matrix is computed successfully
#     if affine_matrix is not None:
#         print("Affine matrix:\n", affine_matrix)
#         return affine_matrix
#     else:
#         print("Affine matrix calculation failed!")
#         return None

# # Function to get affine transformation and initialize match display
# def get_affine_between_sessions(map_reference, gabor_reference):
#     global gray_mapr_ref, gray_gabor_ref, keypoints_ref, keypoints_target

#     # Convert images to grayscale
#     gray_gabor_ref = cv2.cvtColor(gabor_reference, cv2.COLOR_BGR2GRAY)
#     gray_mapr_ref = cv2.cvtColor(map_reference, cv2.COLOR_BGR2GRAY)

#     # Create SIFT detector
#     sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10)
    
#     keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_mapr_ref, None)
#     keypoints_target, descriptors_target = sift.detectAndCompute(gray_gabor_ref, None)

#     # Use BFMatcher to find the best matches between descriptors
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     initial_matches = bf.match(descriptors_ref, descriptors_target)
#     initial_matches = sorted(initial_matches, key=lambda x: x.distance)

#     # Initialize matches
#     global accepted_matches, rejected_matches
#     accepted_matches = initial_matches
#     rejected_matches = []

#     # Create GUI and display matches
#     update_match_display()

#     # Set up mouse callback to allow the user to click on matches
#     cv2.setMouseCallback('Accepted Matches', select_match)
#     cv2.setMouseCallback('Rejected Matches', select_match)

#     # Wait for user to click matches or press 'Esc' to finish
#     while True:
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # ESC key pressed
#             # Calculate the affine matrix after curation
#             global affine_matrix
#             affine_matrix = calculate_affine()
#             break  # Exit the loop

#     cv2.destroyAllWindows()

#     # affine_matrix now contains the result
#     return affine_matrix

# affine_matrix = get_affine_between_sessions(map_reference, gabor_reference)

# # Now affine_matrix contains the result after pressing 'Esc'
# if affine_matrix is not None:
#     print("Final Affine Matrix:")
#     print(affine_matrix)
    
    
#%% MANUALLY MATHIOCNG LOW MAG IMAGES
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


    return affine_matrix

affine_matrix = get_roi_coordinates(
    gray_mapr_ref, gray_gabor_ref, match_to_use=None, eps=10, min_samples=2, 
      angle_deviation_threshold=2, manual=True)
# affine_matrix=gabor_nomag_coordinates

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
#%%
gabor_roi_coordinates = get_roi_coordinates(
    gray_gabor_ref, gray_gabor_roi, match_to_use=None, eps=10, min_samples=2, 
      angle_deviation_threshold=2, manual=True)

# gabor_roi_coordinates = get_roi_coordinates(
#     gray_gabor_ref, gray_gabor_roi, match_to_use=None, eps=10, min_samples=2, 
#       angle_deviation_threshold=2, manual=False)

map_roi_coordinates = get_roi_coordinates(
    gray_mapr_ref, gray_map_roi, match_to_use=10, eps=10, min_samples=2, 
     angle_deviation_threshold=2, manual=False)

gabor_roi_coordinates_registered = cv2.transform(gabor_roi_coordinates.reshape(-1, 1, 2), affine_matrix)

 #%%
# # plt.close('all')    
# def get_roi_coordinates(low_mag, roi, match_to_use=10, eps=10, min_samples=2, angle_deviation_threshold=2):
#     # Create subplots to display images
#     f, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].imshow(low_mag, cmap='gray')
#     ax[0].set_title('Low Magnification Image')
#     ax[1].imshow(roi, cmap='gray')
#     ax[1].set_title('ROI Image')
    
   

#     # Initialize SIFT detector with parameters
#     sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10)

#     # Detect keypoints and descriptors for both images
#     kp_larger, des_larger = sift.detectAndCompute(low_mag, None)
#     kp_smaller, des_smaller = sift.detectAndCompute(roi, None)

#     # Create BFMatcher object (using KNN instead of direct matching)
#     bf = cv2.BFMatcher(cv2.NORM_L2)

#     # Match descriptors using KNN
#     matches = bf.knnMatch(des_smaller, des_larger, k=2)
    
#     # Apply Lowe's ratio test to filter good matches
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 1 * n.distance:  # Lowe's ratio test
#             good_matches.append(m)

#     # Sort matches based on distance (after ratio test)
#     good_matches = sorted(good_matches, key=lambda x: x.distance)

#     # Check if enough good matches are found
#     if len(good_matches) < 5:
#         raise ValueError("Not enough matches to compute homography.")

#     # Extract location of good matches (top matches)
#     top_matches = good_matches[:match_to_use]
#     img_matches = cv2.drawMatches(roi, kp_smaller, low_mag, kp_larger, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     f, ax = plt.subplots()
#     ax.imshow(img_matches)
#     f.suptitle(f'Top {len(matches)} Matches After Clustering, Length and Angle Filtering')
#     plt.show()

#     # Extract positions and compute angles and lengths
#     positions_and_angles = []
#     angles = []
#     lengths = []
#     for m in top_matches:
#         # Get the positions of the matched keypoints
#         pt_smaller = kp_smaller[m.queryIdx].pt  # Point in the ROI (smaller image)
#         pt_larger = kp_larger[m.trainIdx].pt   # Point in the larger image

#         # Calculate the angle between the points
#         dx = pt_larger[0] - pt_smaller[0]
#         dy = pt_larger[1] - pt_smaller[1]
#         angle = np.degrees(np.arctan2(dy, dx))  # Angle in degrees

#         # Calculate the length of the line (distance between the two points)
#         length = np.sqrt(dx**2 + dy**2)

#         # Store the positions, angle, and length
#         positions_and_angles.append((pt_smaller, pt_larger, angle, length))
#         angles.append(angle)
#         lengths.append(length)

#     # Reshape angles for clustering
#     angles = np.array(angles).reshape(-1, 1)

#     # Apply DBSCAN clustering to group lines with similar angles
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(angles)

#     # Find the largest cluster (most frequent label)
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     largest_cluster_label = unique_labels[np.argmax(counts)]

#     # Find the average angle of the largest cluster (mean angle)
#     largest_cluster_angles = [angles[i] for i in range(len(angles)) if labels[i] == largest_cluster_label]
#     mean_angle = np.mean(largest_cluster_angles)

#     # Filter matches that belong to the largest cluster and are within the angular deviation threshold
#     filtered_matches = []
#     filtered_positions_and_angles = []
#     for i, label in enumerate(labels):
#         if label == largest_cluster_label:
#             # Calculate the angular deviation
#             angle_deviation = abs(angles[i] - mean_angle)
#             if angle_deviation <= angle_deviation_threshold:
#                 # Filter based on line length
#                 filtered_matches.append(top_matches[i])
#                 filtered_positions_and_angles.append(positions_and_angles[i])

#     # Draw the matches on the image
#     img_matches = cv2.drawMatches(roi, kp_smaller, low_mag, kp_larger, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
#     f, ax = plt.subplots()
#     ax.imshow(img_matches)
#     f.suptitle(f'Top {len(filtered_matches)} Matches After Clustering, Length and Angle Filtering')
#     plt.show()

#     # Extract points for homography
#     src_pts = np.float32([kp_smaller[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp_larger[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)

#     # Compute homography matrix using the filtered matches
#     H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#     # Define the corners of the smaller image (ROI)
#     h, w = roi.shape[:2]
#     corners_smaller = np.array([
#         [0, 0],
#         [w - 1, 0],
#         [w - 1, h - 1],
#         [0, h - 1]
#     ], dtype='float32').reshape(-1, 1, 2)

#     # Apply homography to get the coordinates in the larger image
#     corners_larger = cv2.perspectiveTransform(corners_smaller, H)

#     # Change shape to (4, 1, 2) before returning
#     corners_larger = corners_larger.reshape(4, 1, 2)

#     # Draw the polygon (corners connected) on the larger image
#     low_mag_with_polygon = low_mag.copy()
#     corners_larger = corners_larger.reshape(-1, 2)  # Flatten to 2D array for drawing
#     pts = corners_larger.astype(np.int32)
#     pts = pts.reshape((-1, 1, 2))

#     # Draw the polygon with a specific color and thickness
#     cv2.polylines(low_mag_with_polygon, [pts], isClosed=True, color=(255, 255, 0), thickness=3)

#     # Plot the polygon on the larger image
#     plt.figure(figsize=(8, 8))
#     plt.imshow(low_mag_with_polygon, cmap='gray')
#     plt.title('Mapped ROI Polygon on Larger Image')
#     plt.show()
    
#     return corners_larger


# gabor_roi_coordinates = get_roi_coordinates(
#     gray_gabor_ref, gray_gabor_roi, match_to_use=None, eps=10, min_samples=2, 
#       angle_deviation_threshold=2
# )


# map_roi_coordinates = get_roi_coordinates(
#     gray_mapr_ref, gray_map_roi, match_to_use=10, eps=10, min_samples=2, 
#      angle_deviation_threshold=2
# )


# # Apply perspective transformation to gabor_roi_coordinates using the homography matrix
# gabor_roi_coordinates_registered = cv2.transform(gabor_roi_coordinates.reshape(-1, 1, 2), affine_matrix)

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


#%% REGISTER TO PROJECTOR DAY

affine_matrix2=get_affine_between_sessions(proj_img,map_reference)
proj_roi_coordinates=get_roi_coordinates(gray_proj_img,gray_proj_roi)
proj_mask_full_fov=get_translated_mask_coordinates(proj_roi_coordinates.reshape(-1, 1, 2), gray_proj_mask)



reregistered_gabor_roi_coordinates=  cv2.transform(gabor_roi_coordinates_registered, affine_matrix2)
reregistered_gabor_mask_full_fov= cv2.transform(registered_gabor_mask_full_fov.astype(np.float32), affine_matrix2)
registered_map_roi_coordinates= cv2.transform(map_roi_coordinates.reshape(-1, 1, 2), affine_matrix2)
registered_map_mask_full_fov= cv2.transform(map_mask_full_fov.astype(np.float32), affine_matrix2)

#%%

plt.figure(figsize=(12, 6))

# Plot Target Image with ROI and Mask Outline
gabor_reference_with_roi = gabor_reference.copy()
# cv2.polylines(gabor_reference_with_roi, [np.int32(gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(gabor_reference_with_roi, [gabor_mask_full_fov], -1, (255, 0, 0), 2)
plt.subplot(1, 3, 1)
plt.title("Target Image with ROI and Mask")
plt.imshow(cv2.cvtColor(gabor_reference_with_roi, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Plot Reference Image with Transformed Mask Outline
map_reference_with_mask = map_reference.copy()
# cv2.polylines(map_reference_with_mask, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(map_reference_with_mask, [np.int32(registered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)
# cv2.polylines(map_reference_with_mask, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(map_reference_with_mask, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)
plt.subplot(1, 3, 2)
plt.title("Reference Image with Transformed Mask")
plt.imshow(cv2.cvtColor(map_reference_with_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')


proj_img_with_mask = proj_img.copy()

# cv2.polylines(proj_img_with_mask, [np.int32(reregistered_gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(proj_img_with_mask, [np.int32(reregistered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)
# cv2.polylines(proj_img_with_mask, [np.int32(registered_map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(proj_img_with_mask, [np.int32(registered_map_mask_full_fov)], -1, (255, 0, 255), 2)
# cv2.polylines(proj_img_with_mask, [np.int32(proj_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)

for rect in proj_mask_full_fov:
    cv2.drawContours(proj_img_with_mask, [np.int32(rect)], -1, (255, 0, 255), 2)

plt.subplot(1, 3, 3)
plt.title("Reference Image with Transformed Mask")
plt.imshow(cv2.cvtColor(proj_img_with_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')


plt.tight_layout()
plt.show()
#%% MAP PROJECTOR TO SAMPLE

#load projected image ref
proj_ref_mask = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\Full3GbaorMap\projected_ref.jpg')
gary_proj_ref_mask = cv2.cvtColor(proj_ref_mask, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(gary_proj_ref_mask, 200, 255, cv2.THRESH_BINARY)
 
source_coords, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# Assuming the mask has one contour (outline)

source_coords=[i.astype('float32') for i in source_coords]
dest_coords = [i.astype('float32') for i in proj_mask_full_fov]




# THIS GETS THE TRANSFORMATION MATRIX BETEWEN SAMPLE SAPCE AND PROJECTOR SPACE
# Compute the homography matrix
H, _ = cv2.findHomography(np.vstack(source_coords), np.vstack(dest_coords))
# Compute the inverse of the homography matrix
H_inv = np.linalg.inv(H)


# # Ensure you have at least 3 points for affine transformation
# if len(box) < 3 or len(source_coords) < 3:
#     raise ValueError("Not enough points to compute affine transformation")

# # Select the first 3 matches for affine transformation
# src_pts = source_coords[:3]
# dst_pts = box[:3]

# # Compute the affine transformation matrix
# homography = cv2.getAffineTransform(src_pts, dst_pts)

    




# Print the homography matrix
print("Homography Matrix:")
print(H)

# THIS CONVERTS GABOR AND V1 MASK TO PROJECOT RESPACE
new_roi_coords_dest_gabor = reregistered_gabor_mask_full_fov.astype('float32').reshape(-1, 1, 2)
new_roi_coords_dest=registered_map_mask_full_fov.astype('float32').reshape(-1, 1, 2)


# Apply the inverse homography to the new ROI coordinates
new_roi_coords_src = cv2.perspectiveTransform(new_roi_coords_dest, H_inv)
new_roi_coords_src_gabor = cv2.perspectiveTransform(new_roi_coords_dest_gabor, H_inv)



#%% PLOTING THE PROJECTOR VS IMAGE SPACES

transformed_coords = cv2.perspectiveTransform(np.array([source_coords]), H)[0]



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

#%% THIS CREATES THE REGISTERED MASKS 
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
#%% THIS  CREATES THE OVERLAPING MASK TO LOAD ONTO PROJECTOR
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the larger and smaller masks
# Ensure these masks are loaded as binary images (0s and 255s)
larger_mask = binary_mask


smaller_mask =binary_mask_gabor


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

# plt.subplot(1, 3, 2)
# plt.title("Larger Mask")
# plt.imshow(larger_mask, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 1)
# plt.title("Smaller Mask")
# plt.imshow(smaller_mask, cmap='gray')
# plt.axis('off')

plt.title("New Mask (Larger - Smaller)")
plt.imshow(new_mask, cmap='gray')
plt.axis('off')

plt.show()

# Save the new mask as a JPEG file (optional)
cv2.imwrite('all_v1_but_gabor.jpg', new_mask)

