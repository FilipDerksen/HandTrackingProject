"""
Minimal Hand Tracking Example using MediaPipe.

This script demonstrates basic hand detection and landmark extraction
without using the HandDetector class.
"""

import cv2
import mediapipe as mp
import time


def main():
    """
    Main function for minimal hand tracking demonstration.
    
    This function shows how to use MediaPipe directly for hand detection
    and landmark extraction with FPS counter.
    """
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe hands solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Initialize timing variables for FPS calculation
    previous_time = 0
    current_time = 0

    while True:
        # Read frame from camera
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            break
            
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Process detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and print landmark positions
                for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                    height, width, channels = image.shape
                    center_x = int(landmark.x * width)
                    center_y = int(landmark.y * height)
                    print(f"Landmark {landmark_id}: ({center_x}, {center_y})")

                # Draw hand landmarks and connections
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Draw FPS counter on image
        cv2.putText(image, str(int(fps)), (10, 70), 
                   cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display the image
        cv2.imshow("Minimal Hand Tracking", image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()