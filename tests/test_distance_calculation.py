"""
Test suite specifically for distance calculation functionality.

This module focuses on testing the find_distance method and related
distance calculation edge cases.
"""

import unittest
import math
import numpy as np
import HandTrackingModule as htm


class TestDistanceCalculation(unittest.TestCase):
    """
    Test cases specifically for distance calculation functionality.
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.detector = htm.HandDetector()
    
    def test_basic_distance_calculation(self):
        """
        Test basic distance calculation with simple coordinates.
        """
        point1 = [0, 0]
        point2 = [3, 4]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        # Test distance calculation (3-4-5 triangle)
        self.assertAlmostEqual(distance, 5.0, places=5)
        self.assertEqual(coords, [0, 0, 3, 4])
    
    def test_landmark_format_distance(self):
        """
        Test distance calculation with landmark format [id, x, y].
        """
        point1 = [4, 100, 200]  # Thumb tip
        point2 = [8, 150, 180]  # Index finger tip
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((150 - 100)**2 + (180 - 200)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [100, 200, 150, 180])
    
    def test_zero_distance(self):
        """
        Test distance calculation when points are identical.
        """
        point = [50, 100]
        distance, _, coords = self.detector.find_distance(point, point, None, False)
        
        self.assertEqual(distance, 0.0)
        self.assertEqual(coords, [50, 100, 50, 100])
    
    def test_negative_coordinates(self):
        """
        Test distance calculation with negative coordinates.
        """
        point1 = [-10, -20]
        point2 = [10, 20]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((10 - (-10))**2 + (20 - (-20))**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [-10, -20, 10, 20])
    
    def test_floating_point_coordinates(self):
        """
        Test distance calculation with floating point coordinates.
        """
        point1 = [1.5, 2.5]
        point2 = [4.5, 6.5]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((4.5 - 1.5)**2 + (6.5 - 2.5)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [1.5, 2.5, 4.5, 6.5])
    
    def test_mixed_point_formats(self):
        """
        Test distance calculation with mixed point formats.
        """
        point1 = [4, 100, 200]  # [id, x, y] format
        point2 = [150, 180]      # [x, y] format
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt((150 - 100)**2 + (180 - 200)**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
        self.assertEqual(coords, [100, 200, 150, 180])
    
    def test_very_small_distance(self):
        """
        Test distance calculation with very small distances.
        """
        point1 = [0.0, 0.0]
        point2 = [0.001, 0.001]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt(0.001**2 + 0.001**2)
        self.assertAlmostEqual(distance, expected_distance, places=10)
    
    def test_very_large_distance(self):
        """
        Test distance calculation with very large distances.
        """
        point1 = [0, 0]
        point2 = [1000000, 1000000]
        
        distance, _, coords = self.detector.find_distance(point1, point2, None, False)
        
        expected_distance = math.sqrt(1000000**2 + 1000000**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)
    
    def test_distance_with_image_drawing(self):
        """
        Test distance calculation with image drawing.
        """
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
        self.assertEqual(coords, [100, 200, 300, 400])
    
    def test_distance_without_drawing(self):
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
    
    def test_distance_no_image(self):
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
        self.assertEqual(coords, [0, 0, 5, 12])


if __name__ == '__main__':
    unittest.main(verbosity=2)
