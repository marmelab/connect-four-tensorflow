from connectfour.game import GAME_STATUS

import numpy as np


class Human:

    def __init__(self, player):
        self.player = player

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def next_move(self, game):
        print(np.matrix(game.board).transpose())
        try:
            best_move = int(
                raw_input(
                    '[' + str(self.player) + '] Which column do you want to play:')
            )
        except ValueError:
            print('Column has to be a number')
            return self.next_move(game)
        return best_move - 1

    def turn_feedback(self, player, column):
        who = 'You'
        if self.player != player:
            who = 'Opponent'
        print('[' + str(self.player) + '] ' + who + ' played column ' +
              str(column + 1))

    def game_feedback(self, game, status, winner):
        print(np.matrix(game.board).transpose())
        if winner == self.player:
            print('[' + str(self.player) + '] You win !')
        else:
            print('[' + str(self.player) + '] Opponent wins !')
