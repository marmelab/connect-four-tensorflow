import unittest
from tool.array_consecutive import number_of_consecutive_values, number_of_consecutive_values_2d


class TestArrayConsecutive(unittest.TestCase):

    def test_it_should_return_true_with_right_amount_of_consecutive_values(self):
        array = [[0, 1, 2, 3], [3, 3, 3, 4], [6, 6, 6, 6]]
        consecutives = number_of_consecutive_values_2d(array)
        self.assertEqual(consecutives[6], 1)

    def test_it_should_return_the_right_value_which_was_consecutive(self):
        array = [[0, 1, 2, 3], [3, 3, 3, 3], [8, 6, 6, 6]]
        consecutives = number_of_consecutive_values_2d(array)
        self.assertEqual(consecutives[3], 1)

    def test_it_should_true_even_with_smaller_consecutive_values_needed(self):
        array = [[0, 1, 2, 0], [3, 3, 2, 4], [6, 8, 2, 9]]
        consecutives = number_of_consecutive_values_2d(array, 2)
        self.assertEqual(consecutives[3], 1)

    def test_it_should_not_consider_consecutive_values_on_different_chunks(self):
        array = [[0, 1, 2, 3], [3, 3, 3, 4], [1, 2, 3, 4]]
        consecutives = number_of_consecutive_values_2d(array)
        self.assertEqual(len(consecutives), 0)

    def test_it_should_return_twice_consecutives_if_it_actually_appears_twice(self):
        array = [[0, 1, 2, 0], [3, 3, 2, 4], [6, 3, 3, 9]]
        consecutives = number_of_consecutive_values_2d(array, 2)
        self.assertEqual(consecutives[3], 2)
