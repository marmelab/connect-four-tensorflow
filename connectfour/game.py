from random import random, randint
import tensorflow as tf

from tool.array_consecutive import number_of_consecutive_values_2d as number_of_consecutive_values
from tool.array_transposer import transpose_horizontally, transpose_diagonally


GAME_STATUS = {
    'PLAYING': 0,
    'FINISHED': 1
}


def merge_add(array1, array2):
    for key in array2:
        if not key in array1:
            array1[key] = 0
        array1[key] += array2[key]
    return array1


class Game:

    def __init__(self, board_width, board_height):
        self.board_width = board_width
        self.board_height = board_height
        self.reset()

    def reset(self):
        self.board = [[0 for i in range(self.board_height)]
                      for j in range(self.board_width)]
        self.current_player = -1 if random() < 0.5 else 1
        # self.current_player = -1
        self.winner = 0
        self.status = GAME_STATUS['PLAYING']
        self.turn = 0

    def random_action(self):
        column = randint(0, self.board_width - 1)
        while not self.is_move_legal(column):
            column = randint(0, self.board_width - 1)
        return column

    def play(self, column_number, current_player):
        column = self.board[column_number]
        for row_number in range(len(column) - 1, -1, -1):
            if column[row_number] == 0:
                column[row_number] = current_player
                break
        self.turn += 1
        self.switch_players()

    def switch_players(self):
        self.current_player *= -1

    def is_board_full(self):
        return not any(any(cell == 0 for cell in column) for column in self.board)

    def get_status(self):
        consecutive_values = self.has_aligned_discs(4)

        player1_won = -1 in consecutive_values
        player2_won = 1 in consecutive_values

        if player1_won:
            self.winner = -1
        elif player2_won:
            self.winner = 1

        status = GAME_STATUS['FINISHED'] \
            if player1_won or player2_won or self.is_board_full() \
            else GAME_STATUS['PLAYING']

        return status



    def has_aligned_discs(self, number_of_cells):
        consecutive_values = {}

        sub_consecutive_values = number_of_consecutive_values(
            self.board, number_of_cells)
        if -1 in sub_consecutive_values or 1 in sub_consecutive_values:
            return sub_consecutive_values

        sub_consecutive_values = number_of_consecutive_values(
            transpose_horizontally(self.board), number_of_cells)
        if -1 in sub_consecutive_values or 1 in sub_consecutive_values:
            return sub_consecutive_values

        sub_consecutive_values = number_of_consecutive_values(
            transpose_diagonally(self.board, True), number_of_cells)
        if -1 in sub_consecutive_values or 1 in sub_consecutive_values:
            return sub_consecutive_values

        sub_consecutive_values = number_of_consecutive_values(
            transpose_diagonally(self.board), number_of_cells)
        if -1 in sub_consecutive_values or 1 in sub_consecutive_values:
            return sub_consecutive_values

        return {}

    def count_aligned_discs(self, number_of_cells, player = None):
        consecutive_values = {}

        sub_consecutive_values = number_of_consecutive_values(
            self.board, number_of_cells, player)
        consecutive_values = merge_add(
            consecutive_values, sub_consecutive_values)

        sub_consecutive_values = number_of_consecutive_values(
            transpose_horizontally(self.board), number_of_cells, player)
        consecutive_values = merge_add(
            consecutive_values, sub_consecutive_values)

        sub_consecutive_values = number_of_consecutive_values(
            transpose_diagonally(self.board, True), number_of_cells, player)
        consecutive_values = merge_add(
            consecutive_values, sub_consecutive_values)

        sub_consecutive_values = number_of_consecutive_values(
            transpose_diagonally(self.board), number_of_cells, player)
        consecutive_values = merge_add(
            consecutive_values, sub_consecutive_values)

        if player == None:
            return consecutive_values

        return {} if not player in consecutive_values else consecutive_values

    def is_move_legal(self, column):
        return self.board[column][0] == 0

    def get_legal_moves(self):
        legal_moves = []
        for column_number in range(self.board_width):
            legal_moves.append(self.is_move_legal(column_number))
        return legal_moves
