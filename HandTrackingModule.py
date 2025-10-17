"""
Hand Tracking Module using MediaPipe.

This module provides a class for detecting and tracking hands in real-time video
using MediaPipe's hand detection solution.
"""

import cv2
import mediapipe as mp
import time


class HandDetector:
    """
    A class for detecting and tracking hands in video frames using MediaPipe.
    
    This class provides methods to detect hand landmarks and draw them on
    video frames with customizable detection parameters.
    """
    
    def __init__(self, static_image_mode=False, max_hands=2, 
                 detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand detector with specified parameters.
        
        Args:
            static_image_mode (bool): Whether to treat input as static images.
            max_hands (int): Maximum number of hands to detect (1-2).
            detection_confidence (float): Minimum confidence for hand detection (0.0-1.0).
            tracking_confidence (float): Minimum confidence for hand tracking (0.0-1.0).
        """
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils


    def find_hands(self, image, draw=True):
        """
        Detect hands in the given image and optionally draw landmarks.
        
        Args:
            image: Input image frame (BGR format).
            draw (bool): Whether to draw hand landmarks on the image.
            
        Returns:
            image: Image with hand landmarks drawn (if draw=True).
        """
        # Convert BGR to RGB for MediaPipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        # Draw hand landmarks if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

        return image
    
    def find_position(self, image, hand_number=0, draw=True):
        """
        Extract landmark positions for a specific hand.
        
        Args:
            image: Input image frame.
            hand_number (int): Index of the hand to track (0 for first hand).
            draw (bool): Whether to draw circles at landmark positions.
            
        Returns:
            list: List of landmark positions [[id, x, y], ...].
        """
        landmark_list = []

        if self.results.multi_hand_landmarks:
            # Get the specified hand
            hand_landmarks = self.results.multi_hand_landmarks[hand_number]

            # Extract landmark positions
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channels = image.shape
                # Convert normalized coordinates to pixel coordinates
                center_x = int(landmark.x * width)
                center_y = int(landmark.y * height)
                
                landmark_list.append([landmark_id, center_x, center_y])
                
                # Draw circle at landmark position if requested
                if draw:
                    cv2.circle(image, (center_x, center_y), 7, (255, 0, 0), cv2.FILLED)

        return landmark_list
 
