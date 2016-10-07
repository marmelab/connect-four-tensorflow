from random import random
import numpy as np
import sys
import getopt
import string

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

    p1_type = 'neural_network'
    p2_type = 'neural_network'
    number_of_games = 1
    randomness = 0.25
    board_width = 4
    board_height = 4

    try:
        opts, args = getopt.getopt(argv, "hv", [
                                   "p1=", "p2=", "board_width=", "board_height=", "iterations=", "randomness="])
    except getopt.GetoptError:
        print 'train.py --p1 <1st player type> --p2 <2nd player type>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'train.py --p1 <1st player type> --p2 <2nd player type>'
            sys.exit()
        if opt == '-v':
            global verbose
            verbose = True
        elif opt in ("--p1"):
            p1_type = arg
        elif opt in ("--p2"):
            p2_type = arg
        elif opt in ("--board_width"):
            board_width = string.atoi(arg)
        elif opt in ("--board_height"):
            board_height = string.atoi(arg)
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

    batch_number = 1

    with ai.factory.create(p1_type, first_player) as p1_ai:
        with ai.factory.create(p2_type, second_player) as p2_ai:
            for game_number in range(number_of_games):
                status = game.get_status()

                while status == GAME_STATUS['PLAYING']:
                    current_player = game.current_player
                    current_ai = p1_ai if current_player == first_player \
                        else p2_ai

                    action = current_ai.next_move(game)

                    if random() < randomness and current_player == first_player:
                        printv('random triggered')
                        action = game.random_action()
                    printv(str(current_player) + ' plays ' + str(action))
                    game.play(action, current_player)
                    printv(np.matrix(game.board).transpose())

                    status = game.get_status()

                    p1_ai.turn_feedback(current_player, action)
                    p2_ai.turn_feedback(current_player, action)

                p1_ai.game_feedback(game, status, game.winner)
                p2_ai.game_feedback(game, status, game.winner)

                if(game.winner == first_player):
                    batch_statuses['won_p1'] += 1
                elif game.winner == second_player:
                    batch_statuses['won_p2'] += 1
                else:
                    batch_statuses['draw'] += 1

                game.reset()
                if game_number > 1 and game_number % 100 == 0:
                    print("### BATCH N " + str(batch_number) + " ###")
                    batch_number += 1
                    handle_stats(game_statuses, batch_statuses,
                                 p1_type, p2_type)

if __name__ == "__main__":
    main(sys.argv[1:])
