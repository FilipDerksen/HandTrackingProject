"""
Hand Tracking Game using the HandDetector module.

This script demonstrates how to use the HandDetector class for
hand tracking applications and games.
"""

import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# Hand landmark constants (21 landmarks total: 0-20)
# Fingertip landmarks for easy selection
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20

# Other useful landmarks
WRIST = 0
THUMB_CMC = 1
INDEX_FINGER_MCP = 5
MIDDLE_FINGER_MCP = 9
RING_FINGER_MCP = 13
PINKY_MCP = 17

# Total number of landmarks available
TOTAL_LANDMARKS = 21


def print_landmark_info():
    """
    Print information about available hand landmarks.
    
    This function displays all 21 hand landmarks with their IDs and names
    for easy reference when developing hand tracking applications.
    """
    print("=== Hand Landmarks Reference ===")
    print(f"Total landmarks available: {TOTAL_LANDMARKS} (0-{TOTAL_LANDMARKS-1})")
    print("\nFingertip landmarks:")
    print(f"  THUMB_TIP = {THUMB_TIP}")
    print(f"  INDEX_FINGER_TIP = {INDEX_FINGER_TIP}")
    print(f"  MIDDLE_FINGER_TIP = {MIDDLE_FINGER_TIP}")
    print(f"  RING_FINGER_TIP = {RING_FINGER_TIP}")
    print(f"  PINKY_TIP = {PINKY_TIP}")
    print("\nOther useful landmarks:")
    print(f"  WRIST = {WRIST}")
    print(f"  THUMB_CMC = {THUMB_CMC}")
    print(f"  INDEX_FINGER_MCP = {INDEX_FINGER_MCP}")
    print(f"  MIDDLE_FINGER_MCP = {MIDDLE_FINGER_MCP}")
    print(f"  RING_FINGER_MCP = {RING_FINGER_MCP}")
    print(f"  PINKY_MCP = {PINKY_MCP}")
    print("\nUsage example:")
    print("  landmark_list[THUMB_TIP]  # Gets thumb tip position")
    print("  landmark_list[INDEX_FINGER_TIP]  # Gets index finger tip position")


def main():
    """
    Main function for hand tracking game demonstration.
    
    This function shows how to use the HandDetector class for
    hand tracking with minimal visual output for game applications.
    """
    # Print landmark reference information
    print_landmark_info()
    print("\nStarting hand tracking game...")
    print("Press 'q' to quit\n")
    
    # Initialize timing variables for FPS calculation
    previous_time = 0
    current_time = 0
    
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    
    # Initialize hand detector
    detector = htm.HandDetector()

    while True:
        # Read frame from camera
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            break
            
        # Detect hands without drawing landmarks (for game performance)
        image = detector.find_hands(image, draw=True)
        landmark_list = detector.find_position(image, draw=False)
        
        # Process hand landmarks if detected
        if len(landmark_list) != 0:
            # ===== EASY LANDMARK SELECTION =====
            # Change the landmark below to track different finger positions
            # Available: THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, 
            #           RING_FINGER_TIP, PINKY_TIP, WRIST, etc.
            # Total landmarks: 0-20 (21 landmarks)
            
            selected_landmark = THUMB_TIP  # Change this to track different landmarks
            print(f"Selected landmark position: {landmark_list[selected_landmark]}")
            
            # Example: Track multiple landmarks simultaneously
            # print(f"Thumb tip: {landmark_list[THUMB_TIP]}")
            # print(f"Index finger tip: {landmark_list[INDEX_FINGER_TIP]}")
            # print(f"Middle finger tip: {landmark_list[MIDDLE_FINGER_TIP]}")

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Draw FPS counter on image
        cv2.putText(image, str(int(fps)), (10, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the image
        cv2.imshow("Hand Tracking Game", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()