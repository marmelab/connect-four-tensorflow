from random import random
import numpy as np
import sys
import getopt
import string
import json
import h5py
import hashlib

from connectfour.game import Game, GAME_STATUS
import ai.factory

verbose = False


def printv(to_print):
    if verbose:
        print(to_print)


def display_stats(game_statuses, p1_type, p2_type):
    total_games = game_statuses['draw'] + \
        game_statuses['won_p1'] + game_statuses['won_p2']
    print('won games p1 (' + p1_type + '): ' + str(game_statuses['won_p1']) + ' / ' + str(
        total_games) + ' (' + str(game_statuses['won_p1'] * 100 / total_games) + '%)')
    print('won games p2 (' + p2_type + '): ' + str(game_statuses['won_p2']) + ' / ' + str(
        total_games) + ' (' + str(game_statuses['won_p2'] * 100 / total_games) + '%)')
    print('draw : ' + str(game_statuses['draw']) + ' / ' + str(
        total_games) + ' (' + str(game_statuses['draw'] * 100 / total_games) + '%)')


def handle_stats(game_statuses, batch_statuses, p1_type, p2_type):
    for key in batch_statuses:
        game_statuses[key] += batch_statuses[key]
    print('--- batch statistics ---')
    display_stats(batch_statuses, p1_type, p2_type)
    print('--- total statistics ---')
    display_stats(game_statuses, p1_type, p2_type)

    for key in batch_statuses:
        batch_statuses[key] = 0


def main(argv):
    first_player = -1
    second_player = 1

    p1_type = 'minimax:2'
    p2_type = 'minimax:1'
    number_of_games = 40000
    randomness = 0.1
    board_width = 7
    board_height = 6

    filename = 'data.h5'
    datafile = h5py.File(filename, "w")

    opponent_level = 2

    try:
        opts, args = getopt.getopt(argv, "hv", [
                                   "opponent=", "board_width=", "board_height=", "iterations=", "o=", "randomness=", "level=", "learn="])
    except getopt.GetoptError:
        print 'generate_data.py --p1 <opponent type> --p2 <opponent type>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'generate_data.py --p1 <opponent type> --p2 <opponent type>'
            sys.exit()
        if opt == '-o':
            filename = arg
        elif opt in ("--p1"):
            p1_type = arg
        elif opt in ("--p2"):
            p2_type = arg
        elif opt in ("--iterations"):
            number_of_games = string.atoi(arg)
        elif opt in ("--randomness"):
            randomness = string.atof(arg)

    game = Game(board_width, board_height)

    batch_statuses = {
        'draw': 0,
        'won_p1': 0,
        'won_p2': 0
    }

    game_statuses = {
        'draw': 0,
        'won_p1': 0,
        'won_p2': 0
    }

    batch_size = 100

    with ai.factory.create(p1_type, first_player) as p1_ai:
        with ai.factory.create(p2_type, second_player) as p2_ai:
            for game_number in range(number_of_games):
                status = game.get_status()
                game_turns = []
                while status == GAME_STATUS['PLAYING']:
                    current_player = game.current_player
                    current_ai = p1_ai if current_player == first_player else p2_ai

                    action = None
                    if random() <= randomness :
                        # Randomness factor is just for exploration,
                        # No saving if random action
                        action = game.random_action()
                    else:
                        action = current_ai.next_move(game)

                        game_turns.append({
                            'player': current_player,
                            'board': game.board,
                            'action': action,
                        })

                    game.play(action, current_player)

                    status = game.get_status()

                if game.winner == first_player:
                    batch_statuses['won_p1'] += 1
                elif game.winner == second_player:
                    batch_statuses['won_p2'] += 1
                else:
                    batch_statuses['draw'] += 1

                won_turns = [turn for turn in game_turns if turn['player'] == game.winner]

                for turn in won_turns:
                    if turn['player'] == -1 :
                        turn['board'] = np.multiply(turn['board'], -1)

                    board_dataset = datafile.create_dataset(turn['board'].__hash__, (board_width, board_height), dtype='i')

                    board_dataset[:] = turn['board'];
                    board_dataset.attrs['action'] = turn['action']

                if (game_number %  batch_size) == 0:
                    display_stats(batch_statuses, p1_type, p2_type)
                    datafile.flush()

                game.reset()
            display_stats(batch_statuses, p1_type, p2_type)

if __name__ == "__main__":
    main(sys.argv[1:])
