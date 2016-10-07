import random
from copy import deepcopy

from connectfour.game import GAME_STATUS


class Minimax:

    def __init__(self, player):
        self.player = player
        self.game = None
        self.colors = [-1, 1]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def next_move(self, game=None):
        if game is None:
            game = self.game

        depth = 2

        opponent_player = self.get_opponent(self.player)

        legal_moves = {}
        for col in range(4):
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

    def turn_feedback(self, player, column):
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

        for i in range(4):
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

        my_fours = game.check_for_streak(color, 4)
        my_threes = game.check_for_streak(color, 3)
        my_twos = game.check_for_streak(color, 2)
        opp_fours = game.check_for_streak(o_color, 4)

        if opp_fours > 0:
            return -100000
        else:
            return my_fours * 100000 + my_threes * 100 + my_twos
