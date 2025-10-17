"""
Distance Calculation Demo

This script demonstrates the find_distance() method of the HandDetector class.
It shows how to calculate distances between hand landmarks and visualize them.
"""

import cv2
import time
import HandTrackingModule as htm

# Hand landmark constants for easy reference
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20


def main():
    """
    Main function demonstrating distance calculation between hand landmarks.
    """
    print("Distance Calculation Demo")
    print("Press 'q' to quit")
    print("This demo shows distances between fingertips")
    print()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()
    
    # Initialize FPS calculator
    previous_time = 0
    
    while True:
        # Read frame from camera
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Detect hands
        image = detector.find_hands(image, draw=True)
        landmark_list = detector.find_position(image, draw=False)
        
        # Calculate and display distances if hand is detected
        if len(landmark_list) != 0:
            # Calculate distance between thumb and index finger
            thumb_pos = landmark_list[THUMB_TIP]
            index_pos = landmark_list[INDEX_FINGER_TIP]
            
            distance, image, coords = detector.find_distance(
                thumb_pos, index_pos, image, draw=True
            )
            
            # Display distance on screen
            cv2.putText(image, f"Thumb-Index Distance: {distance:.1f}px", 
                       (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Calculate distance between index and middle finger
            middle_pos = landmark_list[MIDDLE_FINGER_TIP]
            distance2, image, coords2 = detector.find_distance(
                index_pos, middle_pos, image, draw=True
            )
            
            cv2.putText(image, f"Index-Middle Distance: {distance2:.1f}px", 
                       (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Calculate distance between thumb and pinky
            pinky_pos = landmark_list[PINKY_TIP]
            distance3, image, coords3 = detector.find_distance(
                thumb_pos, pinky_pos, image, draw=True
            )
            
            cv2.putText(image, f"Thumb-Pinky Distance: {distance3:.1f}px", 
                       (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Display landmark information
            cv2.putText(image, f"Landmarks detected: {len(landmark_list)}", 
                       (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        
        cv2.putText(image, f"FPS: {int(fps)}", (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Display the image
        cv2.imshow("Distance Calculation Demo", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
