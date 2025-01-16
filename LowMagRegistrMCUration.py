import cv2
import numpy as np

# Global variables
matches = []
accepted_matches = []
rejected_matches = []
keypoints_ref = []
keypoints_target = []
gray_gabor_ref = None
gray_mapr_ref = None
img_matches = None
affine_matrix = None  # Store the affine matrix globally

# Function to handle mouse click event
def select_match(event, x, y, flags, param):
    global accepted_matches, rejected_matches, img_matches

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click was inside the accepted figure (left)
        if x < img_matches.shape[1] // 2:
            # Determine which match was clicked
            for i, match in enumerate(accepted_matches):
                kp1 = keypoints_ref[match.queryIdx]
                kp2 = keypoints_target[match.trainIdx]
                # Check if the click is near the keypoint
                if abs(kp1.pt[0] - x) < 10 and abs(kp1.pt[1] - y) < 10:
                    # Move match to rejected
                    rejected_matches.append(match)
                    accepted_matches.remove(match)
                    update_match_display()
                    break
        # Check if the click was inside the rejected figure (right)
        else:
            # Determine which match was clicked
            for i, match in enumerate(rejected_matches):
                kp1 = keypoints_ref[match.queryIdx]
                kp2 = keypoints_target[match.trainIdx]
                # Check if the click is near the keypoint
                if abs(kp1.pt[0] - x) < 10 and abs(kp1.pt[1] - y) < 10:
                    # Move match to accepted
                    accepted_matches.append(match)
                    rejected_matches.remove(match)
                    update_match_display()

# Update match display for accepted and rejected matches
def update_match_display():
    global img_matches
    img_matches = np.zeros_like(gray_mapr_ref)

    # Draw accepted matches
    img_matches = cv2.drawMatches(
        gray_mapr_ref, keypoints_ref,  # First image and keypoints
        gray_gabor_ref, keypoints_target,  # Second image and keypoints
        accepted_matches,  # Accepted matches
        img_matches,  # The image where matches are drawn
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Create second figure with rejected matches
    img_rejected_matches = np.zeros_like(gray_mapr_ref)
    img_rejected_matches = cv2.drawMatches(
        gray_mapr_ref, keypoints_ref,  # First image and keypoints
        gray_gabor_ref, keypoints_target,  # Second image and keypoints
        rejected_matches,  # Rejected matches
        img_rejected_matches,  # The image where matches are drawn
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Display the updated figures
    cv2.imshow("Accepted Matches", img_matches)
    cv2.imshow("Rejected Matches", img_rejected_matches)

# Function to calculate affine transformation based on accepted matches
def calculate_affine():
    global accepted_matches

    # Extract points from accepted matches
    points_ref = np.zeros((len(accepted_matches), 2), dtype=np.float32)
    points_target = np.zeros((len(accepted_matches), 2), dtype=np.float32)
    for i, match in enumerate(accepted_matches):
        points_ref[i, :] = keypoints_ref[match.queryIdx].pt
        points_target[i, :] = keypoints_target[match.trainIdx].pt

    # Compute the affine transformation matrix
    affine_matrix, inliers = cv2.estimateAffine2D(points_target, points_ref, method=cv2.RANSAC)

    # If the affine matrix is computed successfully
    if affine_matrix is not None:
        print("Affine matrix:\n", affine_matrix)
        return affine_matrix
    else:
        print("Affine matrix calculation failed!")
        return None

# Function to get affine transformation and initialize match display
def get_affine_between_sessions(map_reference, gabor_reference):
    global gray_mapr_ref, gray_gabor_ref, keypoints_ref, keypoints_target

    # Convert images to grayscale
    gray_gabor_ref = cv2.cvtColor(gabor_reference, cv2.COLOR_BGR2GRAY)
    gray_mapr_ref = cv2.cvtColor(map_reference, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_mapr_ref, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_gabor_ref, None)

    # Use BFMatcher to find the best matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    initial_matches = bf.match(descriptors_ref, descriptors_target)
    initial_matches = sorted(initial_matches, key=lambda x: x.distance)

    # Initialize matches
    global accepted_matches, rejected_matches
    accepted_matches = initial_matches
    rejected_matches = []

    # Create GUI and display matches
    update_match_display()

    # Set up mouse callback to allow the user to click on matches
    cv2.setMouseCallback('Accepted Matches', select_match)
    cv2.setMouseCallback('Rejected Matches', select_match)

    # Wait for user to click matches or press 'Esc' to finish
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key pressed
            # Calculate the affine matrix after curation
            global affine_matrix
            affine_matrix = calculate_affine()
            break  # Exit the loop

    cv2.destroyAllWindows()

    # affine_matrix now contains the result
    return affine_matrix

# Example to load images and get affine transformation
if __name__ == "__main__":
    # Load your images here
    image_folder = r'C:\Users\sp3660\Documents\Projects\LabNY\Amsterdam\Analysis\TempMovies\20241204Issues2'

    # Load images (replace these with actual image loading code)
    map_reference = cv2.imread(image_folder + r'\RecBin_NoMag_Map.jpg')
    gabor_reference = cv2.imread(image_folder + r'\RecBin_NoMag_Gabor.jpg')

    affine_matrix = get_affine_between_sessions(map_reference, gabor_reference)

    # Now affine_matrix contains the result after pressing 'Esc'
    if affine_matrix is not None:
        print("Final Affine Matrix:")
        print(affine_matrix)
