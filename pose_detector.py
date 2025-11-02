import cv2
import mediapipe as mp
import numpy as np
import os



# Core Pose Detector | Detect pose from a region.
def get_pose_landmarks(image_crop, pose_model):
    """
    Runs MediaPipe Pose detection on a single image crop.

    Args:
        image_crop (np.array): The image (as a NumPy array) to process.
        pose_model: The initialized MediaPipe Pose model instance.

    Returns:
        The detected pose_landmarks object, or None if no pose is found.
    """
    # MediaPipe works with RGB, OpenCV uses BGR
    image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    
    # Process the image crop
    results = pose_model.process(image_crop_rgb)
    
    # Return the landmarks
    return results.pose_landmarks

# Bounding Box Loop | Detect all the person and their pose for an image.
def detect_poses_in_boxes(mp_pose, image, bounding_boxes):
    """
    Detects poses within specified bounding boxes in an image.

    Args:
        image (np.array): The full original image.
        bounding_boxes (list): A list of tuples, where each tuple is
                               (x1, y1, x2, y2) defining a bounding box.

    Returns:
        list: A list of tuples, (landmarks, box), where 'landmarks'
              is the detected pose_landmarks object and 'box' is the
              original bounding box it was found in.
    """
    all_pose_data = []
    
    # Initialize the Pose model using a 'with' block for proper management
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            
            # Create the image crop based on the bounding box
            # Add checks for valid box dimensions
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0] or x1 >= x2 or y1 >= y2:
                print(f"Skipping invalid bounding box: {box}")
                continue
                
            image_crop = image[y1:y2, x1:x2]
            
            # Check if crop is empty
            if image_crop.size == 0:
                print(f"Skipping empty crop for box: {box}")
                continue

            # Call the core detector
            landmarks = get_pose_landmarks(image_crop, pose)
            
            if landmarks:
                # If a pose is found, store both the landmarks and the box
                all_pose_data.append((landmarks, box))
                
    return all_pose_data


# Drawing pose to a black image.
def draw_pose_on_image(mp_drawing, mp_pose, LANDMARK_DRAWING_SPEC, CONNECTION_DRAWING_SPEC, original_image_shape, all_pose_data, target_image=None):
    """
    Draws pose skeletons onto a target image or a new black image.

    Args:
        original_image_shape (tuple): The (height, width, channels) of the
                                      original image.
        all_pose_data (list): The list of (landmarks, box) tuples from
                              detect_poses_in_boxes.
        target_image (np.array, optional): The image to draw on. If None,
                                           a new black image is created.

    Returns:
        np.array: The image with poses drawn on it.
    """
    
    h, w, _ = original_image_shape
    
    # 1. Create or prepare the canvas
    if target_image is None:
        # Create a new black image
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Use the provided target image
        # Ensure its size matches the original image shape
        if target_image.shape != original_image_shape:
            print(f"Warning: Target image shape {target_image.shape} does not match original {original_image_shape}.")
            print("Resizing target image to match.")
            canvas = cv2.resize(target_image, (w, h))
        else:
            canvas = target_image.copy() # Use a copy to avoid modifying the original

    # 2. Draw each pose
    for (landmarks, box) in all_pose_data:
        x1, y1, x2, y2 = box
        
        # Create a "view" (a sub-array) of the canvas corresponding to the
        # bounding box. Drawing on this 'canvas_crop' will directly
        # modify the main 'canvas'.
        canvas_crop = canvas[y1:y2, x1:x2]
        
        # Get the dimensions of the crop for drawing
        crop_h, crop_w, _ = canvas_crop.shape
        if crop_h == 0 or crop_w == 0:
            continue # Skip if the crop is invalid

        # Draw the landmarks onto the canvas sub-region
        mp_drawing.draw_landmarks(
            image=canvas_crop,
            landmark_list=landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_DRAWING_SPEC,
            connection_drawing_spec=CONNECTION_DRAWING_SPEC
        )
        
    return canvas