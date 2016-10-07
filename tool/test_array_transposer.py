import unittest
from tool.array_transposer import transpose_horizontally, transpose_diagonally
import json


class TestArrayTransposer(unittest.TestCase):

    def test_it_should_transpose_correctly(self):
        array = [
            [0, 1],
            [2, 1],
        ]

        transposed_array = transpose_horizontally(array)

        expected_array = [
            [0, 2],
            [1, 1],
        ]
        self.assertEqual(json.dumps(transposed_array.tolist()),
                         json.dumps(expected_array))

    def test_it_should_transpose_from_top_left_to_bottom_right_diagonals(self):
        array = [
            [0, 1],
            [2, 1],
        ]

        transposed_array = transpose_diagonally(array)

        expected_array = [
            [0],
            [2, 1],
            [1],
        ]

        self.assertEqual(json.dumps(transposed_array),
                         json.dumps(expected_array))

    def test_it_should_transpose_from_bottom_left_to_top_right_diagonals(self):
        array = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]

        transposed_array = transpose_diagonally(array, True)

        expected_array = [
            [6],
            [7, 3],
            [8, 4, 0],
            [5, 1],
            [2],
        ]

        self.assertEqual(json.dumps(transposed_array),
                         json.dumps(expected_array))
