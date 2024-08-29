import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load images
target_img = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_NoMag_Gabor.jpg')
reference_img = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_NoMag_Map.jpg')
roi_img = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_Roi_Gabor.jpg')
roi2_img= cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_Roi_Map.jpg')
mask_img = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_Roi_Mask_Gabor.png')
mask2_img = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_Roi_Mask_Map.jpg')
proj_img = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_NoMag_Proj.jpg')
proj_roi= cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_Roi_Proj.jpg')
proj_mask = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\RecBin_Roi_Mask_Proj.jpg')
gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
gray_roi2 = cv2.cvtColor(roi2_img, cv2.COLOR_BGR2GRAY)
gray_mask = np.invert(cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY))
gray_mask2 = cv2.cvtColor(mask2_img, cv2.COLOR_BGR2GRAY)
gray_proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2GRAY)
gray_proj_roi = cv2.cvtColor(proj_roi, cv2.COLOR_BGR2GRAY)
gray_proj_mask = cv2.cvtColor(proj_mask, cv2.COLOR_BGR2GRAY)


f,ax=plt.subplots(3,2)
f.tight_layout()
ax.flatten()[0].imshow(gray_target)
ax.flatten()[1].imshow(gray_ref)
ax.flatten()[2].imshow(gray_roi)
ax.flatten()[3].imshow(gray_roi2)
ax.flatten()[4].imshow(gray_mask)
ax.flatten()[5].imshow(gray_mask2)




 
def get_homography_betwen_sessions(reference_img,target_img):
    
    
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_target, None)

    # Use BFMatcher to find the best matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_target)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    points_ref = np.zeros((len(matches), 2), dtype=np.float32)
    points_target = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points_ref[i, :] = keypoints_ref[match.queryIdx].pt
        points_target[i, :] = keypoints_target[match.trainIdx].pt

    # Find the homography matrix
    homography, mask = cv2.findHomography(points_target, points_ref, cv2.RANSAC)
    
    

    # Ensure you have at least 3 points for affine transformation
    # if len(points_ref) < 3 or len(points_target) < 3:
    #     raise ValueError("Not enough points to compute affine transformation")
    
    # # Select the first 3 matches for affine transformation
    # src_pts = points_target[:3]
    # dst_pts = points_ref[:3]
    
    # # Compute the affine transformation matrix
    # homography = cv2.getAffineTransform(src_pts, dst_pts)
    
        

    
    
    height, width = reference_img.shape[:2]
    aligned_img = cv2.warpPerspective(target_img, homography, (width, height))
    
    red_img = np.zeros_like(reference_img)
    green_img = np.zeros_like(target_img)
    blue_img = np.zeros_like(aligned_img)
    
    # Set the color channels
    red_img[:, :, 2] = reference_img[:, :, 2]  # Red channel
    green_img[:, :, 1] = target_img[:, :, 1]    # Green channel
    blue_img[:, :, 0] = aligned_img[:, :, 0]    # Blue channel

    # Convert green_img to RGB and make it the same size as the reference_img
    green_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB)
    green_img = cv2.resize(green_img, (width, height))
    
    # Create the final overlay image
    overlay_img = np.zeros_like(reference_img)
    overlay_img[:, :, 2] = red_img[:, :, 2]    # Red channel
    overlay_img[:, :, 1] = green_img[:, :, 1]  # Green channel
    overlay_img[:, :, 0] = blue_img[:, :, 0]   # Blue channel
    
    
    
    plt.figure(figsize=(12, 8))
    # Show the reference image (in red)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
    plt.title('Reference Image in Red')
    plt.axis('off')
    # Show the overlay image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title('Overlay Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(12, 8))
    # Show the reference image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
    plt.title('Reference Image')
    plt.axis('off')
    # Show the target image (before alignment)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB))
    plt.title('Target Image')
    plt.axis('off')
    # Show the aligned image
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB))
    plt.title('Aligned Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    

    return homography

homography=get_homography_betwen_sessions(reference_img,target_img)

#%%
def get_roi_coordinates(low_mag, roi):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Detect keypoints and descriptors
    kp_larger, des_larger = sift.detectAndCompute(low_mag, None)
    kp_smaller, des_smaller = sift.detectAndCompute(roi, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des_smaller, des_larger)
     
    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    src_pts = np.float32([kp_smaller[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_larger[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    
    # if len(dst_pts) < 3 or len(src_pts) < 3:
    #     raise ValueError("Not enough points to compute affine transformation")
    
    # # Select the first 3 matches for affine transformation
    # src_pts = src_pts[:3]
    # dst_pts = dst_pts[:3]
    
    # # Compute the affine transformation matrix
    # H, _  = cv2.getAffineTransform(src_pts, dst_pts)
    
    
    
    
    
    # Define the corners of the smaller image
    h, w = roi.shape[:2]
    corners_smaller = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype='float32').reshape(-1, 1, 2)
    
    # Apply homography to get the coordinates in the larger image
    corners_larger = cv2.perspectiveTransform(corners_smaller, H)
    return corners_larger

gabor_roi_coordinates=get_roi_coordinates(gray_target,gray_roi)
# Apply the original homography to the ROI to find its position on the reference image
gabor_roi_coordinates_registered = cv2.perspectiveTransform(gabor_roi_coordinates, homography)

map_roi_coordinates=get_roi_coordinates(gray_ref,gray_roi2)


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

gabor_mask_full_fov=get_translated_mask_coordinates(gabor_roi_coordinates, gray_mask)
# Apply the original homography to the ROI to find its position on the reference image
registered_gabor_mask_full_fov= cv2.perspectiveTransform(gabor_mask_full_fov.astype(np.float32), homography)

map_mask_full_fov=get_translated_mask_coordinates(map_roi_coordinates, gray_mask2)



#%%
 # Plot Results
plt.figure(figsize=(12, 6))

# Plot Target Image with ROI and Mask Outline
target_img_with_roi = target_img.copy()
cv2.polylines(target_img_with_roi, [np.int32(gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(target_img_with_roi, [gabor_mask_full_fov], -1, (255, 0, 0), 2)
plt.subplot(1, 2, 1)
plt.title("Target Image with ROI and Mask")
plt.imshow(cv2.cvtColor(target_img_with_roi, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Plot Reference Image with Transformed Mask Outline
reference_img_with_mask = reference_img.copy()
cv2.polylines(reference_img_with_mask, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(reference_img_with_mask, [np.int32(registered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)
cv2.polylines(reference_img_with_mask, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(reference_img_with_mask, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)
plt.subplot(1, 2, 2)
plt.title("Reference Image with Transformed Mask")
plt.imshow(cv2.cvtColor(reference_img_with_mask, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

#%% REGISTER TO PROJECTOR DAY

homography2=get_homography_betwen_sessions(proj_img,reference_img)
proj_roi_coordinates=get_roi_coordinates(gray_proj_img,gray_proj_roi)
proj_mask_full_fov=get_translated_mask_coordinates(proj_roi_coordinates, gray_proj_mask)



reregistered_gabor_roi_coordinates= cv2.perspectiveTransform(gabor_roi_coordinates_registered, homography2)
reregistered_gabor_mask_full_fov= cv2.perspectiveTransform(registered_gabor_mask_full_fov.astype(np.float32), homography2)
registered_map_roi_coordinates= cv2.perspectiveTransform(map_roi_coordinates, homography2)
registered_map_mask_full_fov= cv2.perspectiveTransform(map_mask_full_fov.astype(np.float32), homography2)

#%%

plt.figure(figsize=(12, 6))

# Plot Target Image with ROI and Mask Outline
target_img_with_roi = target_img.copy()
# cv2.polylines(target_img_with_roi, [np.int32(gabor_roi_coordinates)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(target_img_with_roi, [gabor_mask_full_fov], -1, (255, 0, 0), 2)
plt.subplot(1, 3, 1)
plt.title("Target Image with ROI and Mask")
plt.imshow(cv2.cvtColor(target_img_with_roi, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Plot Reference Image with Transformed Mask Outline
reference_img_with_mask = reference_img.copy()
# cv2.polylines(reference_img_with_mask, [np.int32(gabor_roi_coordinates_registered)], True, (0, 255, 0), 2, cv2.LINE_AA)
cv2.drawContours(reference_img_with_mask, [np.int32(registered_gabor_mask_full_fov)], -1, (255, 0, 0), 2)
# cv2.polylines(reference_img_with_mask, [np.int32(map_roi_coordinates)], True, (0, 0, 255), 2, cv2.LINE_AA)
cv2.drawContours(reference_img_with_mask, [np.int32(map_mask_full_fov)], -1, (255, 0, 255), 2)
plt.subplot(1, 3, 2)
plt.title("Reference Image with Transformed Mask")
plt.imshow(cv2.cvtColor(reference_img_with_mask, cv2.COLOR_BGR2RGB))
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
proj_ref_mask = cv2.imread(r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20August\Adjusted\projected_ref.jpg')
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

