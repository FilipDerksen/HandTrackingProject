"""
Test suite for hand detection functionality.

This module tests the find_hands and find_position methods
of the HandDetector class.
"""

import unittest
import numpy as np
import HandTrackingModule as htm


class TestHandDetection(unittest.TestCase):
    """
    Test cases for hand detection functionality.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.detector = htm.HandDetector()
    
    def test_find_hands_with_image(self):
        """
        Test find_hands method with a mock image.
        """
        # Create a mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test find_hands without drawing
        result_image = self.detector.find_hands(mock_image, draw=False)
        
        # Test that image is returned
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, mock_image.shape)
        
        # Test find_hands with drawing
        result_image_draw = self.detector.find_hands(mock_image, draw=True)
        self.assertIsNotNone(result_image_draw)
        self.assertEqual(result_image_draw.shape, mock_image.shape)
    
    def test_find_hands_invalid_image(self):
        """
        Test find_hands method with invalid image input.
        """
        # Test with None
        with self.assertRaises(ValueError):
            self.detector.find_hands(None, draw=False)
        
        # Test with wrong image format
        invalid_image = np.zeros((480, 640), dtype=np.uint8)  # 2D instead of 3D
        with self.assertRaises(ValueError):
            self.detector.find_hands(invalid_image, draw=False)
    
    def test_find_position_no_hands_detected(self):
        """
        Test find_position when no hands are detected.
        """
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Simulate no hands detected by not calling find_hands first
        landmark_list = self.detector.find_position(mock_image, hand_number=0, draw=False)
        
        # Should return empty list when no hands detected
        self.assertEqual(landmark_list, [])
    
    def test_find_position_with_mock_landmarks(self):
        """
        Test find_position with mock landmark data.
        """
        # Create mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create mock results object
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self):
                self.landmark = [MockLandmark(0.5, 0.5) for _ in range(21)]
        
        class MockResults:
            def __init__(self):
                self.multi_hand_landmarks = [MockHandLandmarks()]
        
        # Set mock results
        self.detector.results = MockResults()
        
        # Test find_position
        landmark_list = self.detector.find_position(mock_image, hand_number=0, draw=False)
        
        # Should return 21 landmarks
        self.assertEqual(len(landmark_list), 21)
        
        # Check landmark format [id, x, y]
        for i, landmark in enumerate(landmark_list):
            self.assertEqual(len(landmark), 3)
            self.assertEqual(landmark[0], i)  # ID should match index
            self.assertEqual(landmark[1], 320)  # x = 0.5 * 640
            self.assertEqual(landmark[2], 240)  # y = 0.5 * 480
    
    def test_find_position_different_hand_numbers(self):
        """
        Test find_position with different hand numbers.
        """
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create mock results with multiple hands
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self, offset_x=0.0):
                self.landmark = [MockLandmark(0.5 + offset_x, 0.5) for _ in range(21)]
        
        class MockResults:
            def __init__(self):
                self.multi_hand_landmarks = [
                    MockHandLandmarks(0.0),  # Hand 0
                    MockHandLandmarks(0.1)   # Hand 1
                ]
        
        self.detector.results = MockResults()
        
        # Test hand 0
        landmarks_0 = self.detector.find_position(mock_image, hand_number=0, draw=False)
        self.assertEqual(len(landmarks_0), 21)
        self.assertEqual(landmarks_0[0][1], 320)  # x = 0.5 * 640
        
        # Test hand 1
        landmarks_1 = self.detector.find_position(mock_image, hand_number=1, draw=False)
        self.assertEqual(len(landmarks_1), 21)
        self.assertEqual(landmarks_1[0][1], 384)  # x = 0.6 * 640
    
    def test_find_position_invalid_hand_number(self):
        """
        Test find_position with invalid hand number.
        """
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create mock results with one hand
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self):
                self.landmark = [MockLandmark(0.5, 0.5) for _ in range(21)]
        
        class MockResults:
            def __init__(self):
                self.multi_hand_landmarks = [MockHandLandmarks()]
        
        self.detector.results = MockResults()
        
        # Test with hand number that doesn't exist
        landmarks = self.detector.find_position(mock_image, hand_number=5, draw=False)
        self.assertEqual(landmarks, [])
    
    def test_find_position_edge_coordinates(self):
        """
        Test find_position with edge case coordinates.
        """
        mock_image = np.zeros((100, 200, 3), dtype=np.uint8)  # Small image
        
        # Create mock results with edge coordinates
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        class MockHandLandmarks:
            def __init__(self):
                self.landmark = [
                    MockLandmark(0.0, 0.0),    # Top-left
                    MockLandmark(1.0, 1.0),    # Bottom-right
                    MockLandmark(0.5, 0.5),   # Center
                ] + [MockLandmark(0.5, 0.5) for _ in range(18)]  # Fill to 21
        
        class MockResults:
            def __init__(self):
                self.multi_hand_landmarks = [MockHandLandmarks()]
        
        self.detector.results = MockResults()
        
        landmarks = self.detector.find_position(mock_image, hand_number=0, draw=False)
        
        # Check edge coordinates
        self.assertEqual(landmarks[0][1], 0)   # x = 0.0 * 200
        self.assertEqual(landmarks[0][2], 0)   # y = 0.0 * 100
        self.assertEqual(landmarks[1][1], 200) # x = 1.0 * 200
        self.assertEqual(landmarks[1][2], 100) # y = 1.0 * 100
        self.assertEqual(landmarks[2][1], 100) # x = 0.5 * 200
        self.assertEqual(landmarks[2][2], 50)  # y = 0.5 * 100


if __name__ == '__main__':
    unittest.main(verbosity=2)
