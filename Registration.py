# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:19:26 2024

@author: sp3660
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images_sift_and_overlay(reference_image_path, target_image_path, output_path):
    # Load images
    reference_img = cv2.imread(reference_image_path)
    target_img = cv2.imread(target_image_path)
    
    if reference_img is None or target_img is None:
        print("Error loading images.")
        return

    # Convert to grayscale
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_target, None)

    if keypoints_ref is None or keypoints_target is None:
        print("No keypoints found.")
        return

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

    if homography is None:
        print("Homography computation failed.")
        return

    # Warp the target image to align with the reference image
    height, width = reference_img.shape[:2]
    aligned_img = cv2.warpPerspective(target_img, homography, (width, height))

    # Create color images
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

    # Save the overlay image
    # cv2.imwrite(output_path, overlay_img)
    cv2.imwrite(output_path, aligned_img)

    print(f"Overlay image saved to {output_path}")
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




    # Plot the reference image, target image, and aligned image
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
    
    return homography


def apply_registration(new_image_path, homography_matrix, reference_image_shape, output_path):
    # Load the new image
    new_img = cv2.imread(new_image_path)
    
    if new_img is None:
        print("Error loading new image.")
        return

    # Extract the dimensions of the reference image
    height, width = reference_image_shape[:2]

    # Warp the new image using the pre-computed homography matrix
    aligned_new_img = cv2.warpPerspective(new_img, homography_matrix, (width, height))

    # Save the aligned new image
    cv2.imwrite(output_path, aligned_new_img)
    print(f"Aligned new image saved to {output_path}")

    # Plot the new image and the aligned new image
    plt.figure(figsize=(12, 6))

    # Show the original new image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    plt.title('New Image')
    plt.axis('off')

    # Show the aligned new image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(aligned_new_img, cv2.COLOR_BGR2RGB))
    plt.title('Aligned New Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    

# Example usage
reference_image_path = r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\RetMapRef.jpg'
target_image_path = r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\GaborRef.jpg'
output_image_path = r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\FullRegistered.jpg'


homography_matrix =align_images_sift_and_overlay(reference_image_path, target_image_path, output_image_path)
# output_image_path = 'path/to/save/aligned_new_image.jpg'
