import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define source and destination points (for 1 rectangle)
src_pts = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]], dtype=np.float32)  # Source rectangle

# Define the destination rectangle and scale it down slightly
dst_pts = np.array([[[120, 150]], [[220, 180]], [[230, 270]], [[130, 240]]], dtype=np.float32)  # Original destination rectangle

# Scale down the destination rectangle (for example, by a factor of 0.8)
scale_factor = 0.8
center = np.mean(dst_pts, axis=0)  # Calculate the center of the destination rectangle

# Apply scaling to each point relative to the center
scaled_dst_pts = (dst_pts - center) * scale_factor + center

# Reshape the points to match what `findHomography` expects
src_pts = src_pts.reshape(-1, 2)  # Reshape to (4, 2)
scaled_dst_pts = scaled_dst_pts.reshape(-1, 2)  # Reshape the scaled destination points

# Create a function to draw rectangles
def draw_rectangle(image, points, color=(0, 0, 0), thickness=2):
    points = points.astype(int)
    for i in range(4):
        start_point = tuple(points[i])
        end_point = tuple(points[(i + 1) % 4])
        cv2.line(image, start_point, end_point, color, thickness)

# Create a black background (3-channel black image for colored rectangles)
image_size = (500, 500)  # Define image size (500x500)
image_black = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)  # Black background (3-channel)

# First figure: 3 plots (source rectangle, destination rectangle, and transformed source)
plt.figure(figsize=(15, 5))

# Plot 1: Black image with the red source rectangle
image_plot1 = image_black.copy()
draw_rectangle(image_plot1, src_pts, color=(255, 0, 0))  # Red for source
plt.subplot(1, 3, 1)
plt.imshow(image_plot1)  # Display the image in RGB format
plt.title('Source Coordinates (Red)')
plt.axis('off')

# Plot 2: Black image with the scaled green destination rectangle
image_plot2 = image_black.copy()
draw_rectangle(image_plot2, scaled_dst_pts, color=(0, 255, 0))  # Green for destination (scaled)
plt.subplot(1, 3, 2)
plt.imshow(image_plot2)  # Display the image in RGB format
plt.title('Scaled Destination Coordinates (Green)')
plt.axis('off')

# Calculate the homography matrix using `cv2.findHomography()`
H, _ = cv2.findHomography(src_pts, scaled_dst_pts)

# Plot 3: Red background for the transformed rectangle (only inside of the rectangle)
image_plot3 = image_black.copy()  # Start with the black background

# Apply the forward homography transformation (src -> dst)
src_pts_homogeneous = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])  # Homogeneous coordinates
transformed_pts = np.dot(H, src_pts_homogeneous.T).T  # Apply homography (3x3 matrix * 4x1 points)
transformed_pts /= transformed_pts[:, 2].reshape(-1, 1)  # Normalize by the third coordinate (homogeneous)

# Draw the transformed (forward) rectangle in red (with filled color inside)
transformed_pts_int = transformed_pts[:, :2].astype(int)
cv2.fillPoly(image_plot3, [transformed_pts_int], (255, 0, 0))  # Fill inside of transformed rectangle with red

# Draw the transformed (forward) rectangle border in red
draw_rectangle(image_plot3, transformed_pts[:, :2], color=(255, 0, 0), thickness=2)  # Red for transformed border

# Draw the scaled destination rectangle in green (to overlap the red one)
draw_rectangle(image_plot3, scaled_dst_pts, color=(0, 255, 0))  # Green for scaled destination

# Display the third plot with transformed rectangle filled inside in red
plt.subplot(1, 3, 3)
plt.imshow(image_plot3)  # Display the image in RGB format
plt.title('Forward Transform (Red Inside) and Scaled Destination (Green)')
plt.axis('off')

plt.tight_layout()
plt.show()


# Second figure: Source rectangle in red and backward transformation of destination rectangle in filled green

# Create new figure
plt.figure(figsize=(15, 5))

# Plot 1: Black image with the red source rectangle
image_plot1 = image_black.copy()
draw_rectangle(image_plot1, src_pts, color=(255, 0, 0))  # Red for source
plt.subplot(1, 3, 1)
plt.imshow(image_plot1)  # Display the image in RGB format
plt.title('Source Coordinates (Red)')
plt.axis('off')

# Plot 2: Black image with the scaled green destination rectangle
image_plot2 = image_black.copy()
draw_rectangle(image_plot2, scaled_dst_pts, color=(0, 255, 0))  # Green for destination (scaled)
plt.subplot(1, 3, 2)
plt.imshow(image_plot2)  # Display the image in RGB format
plt.title('Scaled Destination Coordinates (Green)')
plt.axis('off')

# Calculate the inverse homography matrix using `cv2.invert()`
H_inv = np.linalg.inv(H)

# Plot 3: Red background for the transformed rectangle (only inside of the rectangle)
image_plot3 = image_black.copy()  # Start with the black background

# Apply the backward homography transformation (dst -> src)
dst_pts_homogeneous = np.hstack([scaled_dst_pts, np.ones((scaled_dst_pts.shape[0], 1))])  # Homogeneous coordinates
backward_transformed_pts = np.dot(H_inv, dst_pts_homogeneous.T).T  # Apply inverse homography (3x3 matrix * 4x1 points)
backward_transformed_pts /= backward_transformed_pts[:, 2].reshape(-1, 1)  # Normalize by the third coordinate (homogeneous)

# Draw the source (forward) rectangle in red
draw_rectangle(image_plot3, src_pts, color=(255, 0, 0), thickness=2)  # Red for source border

# Fill inside the backward transformed (inverse) rectangle in green
backward_transformed_pts_int = backward_transformed_pts[:, :2].astype(int)
cv2.fillPoly(image_plot3, [backward_transformed_pts_int], (0, 255, 0))  # Fill inside of backward transformed rectangle with green

# Draw the boundary of the backward transformed destination rectangle in green
draw_rectangle(image_plot3, backward_transformed_pts[:, :2], color=(0, 255, 0), thickness=2)  # Green for boundary

# Display the third plot with the red source rectangle and filled green backward transformed rectangle
plt.subplot(1, 3, 3)
plt.imshow(image_plot3)  # Display the image in RGB format
plt.title('Source (Red) and Backward Transform (Green Inside)')
plt.axis('off')

plt.tight_layout()
plt.show()
