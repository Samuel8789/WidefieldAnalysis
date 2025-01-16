# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:38:13 2025

@author: sp3660
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


# Load and preprocess the image
projected_ref_mask = cv2.imread(r'C:\Users\calci\Dropbox\projected_ref.jpg')
gray_projected_ref_mask = cv2.cvtColor(projected_ref_mask, cv2.COLOR_BGR2GRAY)
_, binary_proj = cv2.threshold(gray_projected_ref_mask, 200, 255, cv2.THRESH_BINARY)

# Find contours (source coordinates)
source_coords, _ = cv2.findContours(binary_proj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert contours to float32 and squeeze
source_coords = [i.squeeze().astype('float32') for i in source_coords]
dest_coords = [i.squeeze().astype('float32') for i in proj_mask_full_fov]


# source_coords = [i.squeeze().astype('float32') for i in source_coords[0:1]]
# dest_coords = [i.squeeze().astype('float32') for i in proj_mask_full_fov[0:1]]
# source_coords=[source_coords[1]]+[source_coords[4]]
# dest_coords=[dest_coords[1]]+[dest_coords[4]]


# Function to order rectangle corners consistently
def order_corners(corners):
    """
    Orders the corners of a rectangle in clockwise order starting from the top-left.
    """
    center = np.mean(corners, axis=0)
    ordered = sorted(corners, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
    return np.array(ordered, dtype='float32')

# Order corners for both source and destination coordinates
source_coords = [order_corners(c) for c in source_coords]
dest_coords = [order_corners(c) for c in dest_coords]

# Ensure coordinates are reshaped properly for homography
source_coords_reshaped = [c.reshape(-1, 1, 2) for c in source_coords]
dest_coords_reshaped = [c.reshape(-1, 1, 2) for c in dest_coords]


for rect in source_coords_reshaped:
    assert rect.shape[1:] == (1, 2), f"Invalid shape: {rect.shape}"
for rect in dest_coords_reshaped:
    assert rect.shape[1:] == (1, 2), f"Invalid shape: {rect.shape}"


# Compute homography matrix and its inverse
H, _ = cv2.findHomography(np.vstack(source_coords_reshaped), np.vstack(dest_coords_reshaped))
H_inv = np.linalg.inv(H)

# Print the homography matrix
print("Homography Matrix:")
print(H)

# Visualization: Reference and projection spaces
final_reference = proj_ref.copy()
for rect in proj_mask_full_fov:
    cv2.drawContours(final_reference, [np.int32(rect)], -1, (255, 0, 255), 2)

projected_ref = np.zeros((*gray_projected_ref_mask.shape, 3), dtype='uint8')
for rect in source_coords_reshaped:
    cv2.drawContours(projected_ref, [np.int32(rect)], -1, (255, 0, 255), 2)

# Display the original and processed images
f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(final_reference, cv2.COLOR_BGR2RGB))
ax[0].set_title("Final Reference")
ax[1].imshow(cv2.cvtColor(projected_ref, cv2.COLOR_BGR2RGB))
ax[1].set_title("Projected Reference")
plt.show()

# Verify inverse transformation and visualization
checked_coordinates = [cv2.perspectiveTransform(rect, H_inv) for rect in dest_coords_reshaped]

calculated_ref = np.zeros((*gray_projected_ref_mask.shape, 3), dtype='uint8')
for rect in checked_coordinates:
    cv2.drawContours(calculated_ref, [np.int32(rect)], -1, (255, 0, 255), 2)

# Compare original and reconstructed spaces
f, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(projected_ref, cv2.COLOR_BGR2RGB))
ax[0].set_title("Projected Reference")
ax[1].imshow(cv2.cvtColor(calculated_ref, cv2.COLOR_BGR2RGB))
ax[1].set_title("Calculated Reference (Inverse)")
plt.show()

# Debugging checks
for i, (src, dst) in enumerate(zip(source_coords_reshaped, dest_coords_reshaped)):
    print(f"Source Rectangle {i}: {src}")
    print(f"Destination Rectangle {i}: {dst}")
    transformed = cv2.perspectiveTransform(src, H)
    print(f"Transformed (forward): {transformed}")
    inverse_transformed = cv2.perspectiveTransform(transformed, H_inv)
    print(f"Inverse Transformed: {inverse_transformed}")
    print("-----------")

# Validate coordinate consistency
for i, (src, inv) in enumerate(zip(source_coords_reshaped, [cv2.perspectiveTransform(cv2.perspectiveTransform(s, H), H_inv) for s in source_coords_reshaped])):
    assert np.allclose(src, inv, atol=1e-1), f"Mismatch in coordinates for rectangle {i}"
