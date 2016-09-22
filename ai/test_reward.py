import unittest
import tensorflow as tf

from ai.bitmasks import bitmasks
from ai.reward import board, mask, is_aligned


class TestStringMethods(unittest.TestCase):

    def test_is_aligned_detects_2nd_row(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]

            self.assertEqual(
                session.run(is_aligned(), feed_dict={
                    board: game_board,
                    mask: bitmasks[3]
                }),
                True
            )

    def test_is_aligned_detects_2nd_column(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]

            self.assertEqual(
                session.run(is_aligned(), feed_dict={
                    board: game_board,
                    mask: bitmasks[7]
                }),
                True
            )

    def test_is_aligned_detects_1st_diagonal(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]

            self.assertEqual(
                session.run(is_aligned(), feed_dict={
                    board: game_board,
                    mask: bitmasks[0]
                }),
                True
            )

    def test_is_aligned_do_not_detect_false_positive(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]

            self.assertEqual(
                session.run(is_aligned(), feed_dict={
                    board: game_board,
                    mask: bitmasks[1]
                }),
                False
            )
