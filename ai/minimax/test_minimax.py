import unittest
from minimax import Minimax
from connectfour.game import Game

class TestMiniMax(unittest.TestCase):

    def test_next_move_gives_the_best_defence_move(self):
        minimax = Minimax(1)
        game = Game(4, 4)

        game.board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, -1, -1, -1],
            [0, 0, 1, 1],
        ]

        self.assertEqual(minimax.next_move(game), 2)

    def test_next_move_gives_the_best_attack_move(self):
        minimax = Minimax(1)
        game = Game(4, 4)

        game.board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, -1, -1, -1],
            [0, 1, 1, 1],
        ]

        self.assertEqual(minimax.next_move(game), 3)

    def test_next_move_gives_the_best_move_with_diagonals(self):
        minimax = Minimax(1)
        game = Game(4, 4)

        game.board = [
            [0, -1, 1, -1],
            [0, 1, -1, -1],
            [0, -1, 1, -1],
            [0, 0, 0, 1],
        ]

        self.assertEqual(minimax.next_move(game), 0)
