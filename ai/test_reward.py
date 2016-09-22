import unittest
import tensorflow as tf

from bitmasks import bitmasks
from reward import board, get_board_reward


class TestReward(unittest.TestCase):

    def test_is_aligned_detects_2nd_row(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]

            result = session.run(get_board_reward, feed_dict={
                board: game_board
            })

            self.assertEqual(result, 1)

    def test_is_aligned_detects_2nd_column(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]

            result = session.run(get_board_reward, feed_dict={
                board: game_board,
            })

            self.assertEqual(result, 1)

    def test_is_aligned_detects_1st_diagonal(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]

            result = session.run(get_board_reward, feed_dict={
                board: game_board,
            })

            self.assertEqual(result, 1)

    def test_is_aligned_detects_when_lost(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [2, 2, 2, 2],
            ]

            result = session.run(get_board_reward, feed_dict={
                board: game_board
            })

            self.assertEqual(result, -1)

    def test_is_aligned_detects_when_nothing(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            game_board = [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [2, 1, 2, 2],
            ]

            result = session.run(get_board_reward, feed_dict={
                board: game_board
            })

            self.assertEqual(result, 0)
