import random
from copy import deepcopy
import numpy as np

from connectfour.game import GAME_STATUS


class Minimax:

    def __init__(self, player, level):
        self.player = player
        self.game = None
        self.colors = [-1, 1]
        self.level = level

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def next_move(self, game=None):
        if game is None:
            game = self.game

        depth = self.level

        opponent_player = self.get_opponent(self.player)

        legal_moves = {}
        for col in range(game.board_width):
            if game.is_move_legal(col):
                game_copy = deepcopy(game)
                game_copy.play(col, self.player)
                legal_moves[col] = - \
                    self.search(depth - 1, opponent_player, game_copy)

        best_alpha = -99999999
        best_move = None
        moves = legal_moves.items()
        random.shuffle(list(moves))

        for move, alpha in moves:
            if alpha >= best_alpha:
                best_alpha = alpha
                best_move = move

        return best_move

    def turn_feedback(self, game, column):
        pass

    def game_feedback(self, game, status, winner):
        pass

    def get_opponent(self, current_player):
        if current_player == self.colors[0]:
            return self.colors[1]
        else:
            return self.colors[0]

    def search(self, depth, current_player, game=None):
        legal_moves = []
        if game is None:
            game = self.game

        for i in range(game.board_width):
            if game.is_move_legal(i):
                game_copy = deepcopy(game)
                game_copy.play(i, current_player)
                legal_moves.append(game_copy)

        if depth == 0 or len(legal_moves) == 0 or game.get_status() == GAME_STATUS['FINISHED']:
            return self.value(current_player, game)

        opponent_player = self.get_opponent(current_player)

        alpha = -99999999
        for child in legal_moves:
            alpha = max(alpha, -self.search(depth - 1, opponent_player, child))
        return alpha

    def value(self, color, game=None):
        if game is None:
            game = self.game

        o_color = self.get_opponent(color)

        aligned_fours = game.count_aligned_discs(4)
        aligned_threes = game.count_aligned_discs(3, color)
        aligned_twos = game.count_aligned_discs(2, color)
        my_fours = 0
        my_threes = 0
        my_twos = 0
        opp_fours = 0
        
        if color in aligned_fours:
            my_fours = aligned_fours[color]
        if color in aligned_threes:
            my_threes = aligned_threes[color]
        if color in aligned_twos:
            my_twos = aligned_twos[color]
        if o_color in aligned_fours:
            opp_fours = aligned_fours[o_color]

        if opp_fours > 0:
            return -100000
        else:
            return my_fours * 100000 + my_threes * 100 + my_twos
