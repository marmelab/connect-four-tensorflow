import unittest
from game import Game


class TestGame(unittest.TestCase):

    def test_is_move_legal_should_return_true_when_board_empty(self):
        game = Game(4, 4)

        self.assertEqual(game.is_move_legal(0), True)
        self.assertEqual(game.is_move_legal(1), True)
        self.assertEqual(game.is_move_legal(2), True)
        self.assertEqual(game.is_move_legal(3), True)

    def test_is_move_legal_should_return_true_when_played_once(self):
        game = Game(4, 4)

        game.play(0, 1)

        self.assertEqual(game.is_move_legal(0), True)
        self.assertEqual(game.is_move_legal(1), True)
        self.assertEqual(game.is_move_legal(2), True)
        self.assertEqual(game.is_move_legal(3), True)

    def test_is_move_legal_should_return_false_when_column_full(self):
        game = Game(4, 4)
        game.board = [
            [1, -1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1],
        ]

        self.assertEqual(game.is_move_legal(0), False)
