"""
Test suite for HandDetector class methods.

This module contains unit tests for the HandDetector class, focusing on
the distance calculation method and other core functionality.
"""

import unittest
import math
import numpy as np
import HandTrackingModule as htm


class TestHandDetector(unittest.TestCase):
    """
    Test cases for HandDetector class methods.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.detector = htm.HandDetector()
    
    def test_find_distance_basic(self):
        """
        Test basic distance calculation with simple coordinates.
        """
        point1 = [0, 0]
        point2 = [3, 4]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        # Test distance calculation (3-4-5 triangle)
        self.assertAlmostEqual(distance, 5.0, places=5)
        
        # Test coordinate extraction
        self.assertEqual(coords, [0, 0, 3, 4])
    
    def test_find_distance_landmark_format(self):
        """
        Test distance calculation with landmark format [id, x, y].
        """
        point1 = [4, 100, 200]  # Thumb tip
        point2 = [8, 150, 180]  # Index finger tip
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        # Calculate expected distance
        expected_distance = math.sqrt((150 - 100)**2 + (180 - 200)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        
        # Test coordinate extraction
        self.assertEqual(coords, [100, 200, 150, 180])
    
    def test_find_distance_zero_distance(self):
        """
        Test distance calculation when points are identical.
        """
        point1 = [50, 100]
        point2 = [50, 100]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        self.assertEqual(distance, 0.0)
        self.assertEqual(coords, [50, 100, 50, 100])
    
    def test_find_distance_negative_coordinates(self):
        """
        Test distance calculation with negative coordinates.
        """
        point1 = [-10, -20]
        point2 = [10, 20]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((10 - (-10))**2 + (20 - (-20))**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [-10, -20, 10, 20])
    
    def test_find_distance_large_numbers(self):
        """
        Test distance calculation with large coordinate values.
        """
        point1 = [0, 0]
        point2 = [1000, 1000]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt(1000**2 + 1000**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [0, 0, 1000, 1000])
    
    def test_find_distance_floating_point(self):
        """
        Test distance calculation with floating point coordinates.
        """
        point1 = [1.5, 2.5]
        point2 = [4.5, 6.5]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((4.5 - 1.5)**2 + (6.5 - 2.5)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [1.5, 2.5, 4.5, 6.5])
    
    def test_find_distance_with_image_drawing(self):
        """
        Test distance calculation with image drawing (mock test).
        """
        # Create a mock image (numpy array)
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        point1 = [100, 200]
        point2 = [300, 400]
        
        distance, returned_image, coords = self.detector.find_distance(
            point1, point2, mock_image, True
        )
        
        # Test that image is returned
        self.assertIsNotNone(returned_image)
        self.assertEqual(returned_image.shape, mock_image.shape)
        
        # Test distance calculation
        expected_distance = math.sqrt((300 - 100)**2 + (400 - 200)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        
        # Test coordinates
        self.assertEqual(coords, [100, 200, 300, 400])
    
    def test_find_distance_no_drawing(self):
        """
        Test distance calculation without drawing.
        """
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        point1 = [50, 75]
        point2 = [150, 175]
        
        distance, returned_image, coords = self.detector.find_distance(
            point1, point2, mock_image, False
        )
        
        # Test that image is still returned
        self.assertIsNotNone(returned_image)
        
        # Test distance calculation
        expected_distance = math.sqrt((150 - 50)**2 + (175 - 75)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
    
    def test_find_distance_no_image(self):
        """
        Test distance calculation without providing an image.
        """
        point1 = [0, 0]
        point2 = [5, 12]
        
        distance, returned_image, coords = self.detector.find_distance(
            point1, point2, None, True
        )
        
        # Test distance calculation (5-12-13 triangle)
        self.assertAlmostEqual(distance, 13.0, places=5)
        
        # Test that no image is returned
        self.assertIsNone(returned_image)
        
        # Test coordinates
        self.assertEqual(coords, [0, 0, 5, 12])
    
    def test_detector_initialization(self):
        """
        Test HandDetector initialization with default parameters.
        """
        detector = htm.HandDetector()
        
        self.assertFalse(detector.static_image_mode)
        self.assertEqual(detector.max_hands, 2)
        self.assertEqual(detector.detection_confidence, 0.5)
        self.assertEqual(detector.tracking_confidence, 0.5)
    
    def test_detector_initialization_custom_params(self):
        """
        Test HandDetector initialization with custom parameters.
        """
        detector = htm.HandDetector(
            static_image_mode=True,
            max_hands=1,
            detection_confidence=0.8,
            tracking_confidence=0.9
        )
        
        self.assertTrue(detector.static_image_mode)
        self.assertEqual(detector.max_hands, 1)
        self.assertEqual(detector.detection_confidence, 0.8)
        self.assertEqual(detector.tracking_confidence, 0.9)


class TestDistanceCalculationEdgeCases(unittest.TestCase):
    """
    Test edge cases for distance calculation.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.detector = htm.HandDetector()
    
    def test_find_distance_very_small_distance(self):
        """
        Test distance calculation with very small distances.
        """
        point1 = [0.0, 0.0]
        point2 = [0.001, 0.001]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt(0.001**2 + 0.001**2)
        self.assertAlmostEqual(distance, expected_distance, places=10)
    
    def test_find_distance_very_large_distance(self):
        """
        Test distance calculation with very large distances.
        """
        point1 = [0, 0]
        point2 = [1000000, 1000000]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt(1000000**2 + 1000000**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
    
    def test_find_distance_mixed_formats(self):
        """
        Test distance calculation with mixed point formats.
        """
        point1 = [4, 100, 200]  # [id, x, y] format
        point2 = [150, 180]      # [x, y] format
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((150 - 100)**2 + (180 - 200)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [100, 200, 150, 180])


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
