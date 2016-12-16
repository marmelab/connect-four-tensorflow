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

    p1_type = 'keras'
    p2_type = 'minimax'
    number_of_games = 1
    randomness = 0.1
    board_width = 7
    board_height = 6

    opponent_level = 2
    learn = False

    try:
        opts, args = getopt.getopt(argv, "hv", [
                                   "opponent=", "board_width=", "board_height=", "iterations=", "randomness=", "level=", "learn="])
    except getopt.GetoptError:
        print 'train.py --opponent <opponent type>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'train.py --opponent <opponent type>'
            sys.exit()
        if opt == '-v':
            global verbose
            verbose = True
        elif opt in ("--opponent"):
            p2_type = arg
        elif opt in ("--level"):
            opponent_level = string.atoi(arg)
        elif opt in ("--board_width"):
            board_width = string.atoi(arg)
        elif opt in ("--board_height"):
            board_height = string.atoi(arg)
        elif opt in ("--iterations"):
            number_of_games = string.atoi(arg)
        elif opt in ("--randomness"):
            randomness = string.atof(arg)
        elif opt in ("--learn"):
            learn = True if arg == "true" else False

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

    with ai.factory.create(p1_type, first_player, learn = learn) as neural_network_ai:
        with ai.factory.create(p2_type, second_player, level = opponent_level) as opponent_ai:
            for game_number in range(number_of_games):
                status = game.get_status()

                while status == GAME_STATUS['PLAYING']:
                    current_player = game.current_player

                    action = None
                    opponent_action = None

                    if current_player == first_player :
                        # neural network
                        action = neural_network_ai.next_move(game)
                        if p2_type != 'human':
                            opponent_action = opponent_ai.next_move(game)
                        printv('Neural network plays ' + str(action) + '(' + str(opponent_action) + ')')

                        if learn:
                            neural_network_ai.turn_feedback(game, action, opponent_action)
                    else :
                        # opponent
                        action = opponent_ai.next_move(game)
                        opponent_action = neural_network_ai.next_move(game)
                        printv(p2_type + ' plays ' + str(action) + '(' + str(opponent_action) + ')')

                        if random() <= randomness :
                            action = game.random_action()
                        elif learn:
                            neural_network_ai.opponent_turn_feedback(game, opponent_action, action)

                    game.play(action, current_player)
                    if p2_type != 'human':
                        printv(np.matrix(game.board).transpose())

                    status = game.get_status()

                    opponent_ai.turn_feedback(current_player, action)

                printv(('Neural network' if game.winner == -1 else p2_type) + ' wins')
                neural_network_ai.game_feedback(game, status, game.winner)
                opponent_ai.game_feedback(game, status, game.winner)

                if(game.winner == first_player):
                    batch_statuses['won_p1'] += 1
                elif game.winner == second_player:
                    batch_statuses['won_p2'] += 1
                else:
                    batch_statuses['draw'] += 1

                game.reset()
                if game_number > 1 and game_number % 10 == 0:
                    print("### BATCH N " + str(batch_number) + " ###")
                    batch_number += 1
                    handle_stats(game_statuses, batch_statuses,
                                 p1_type, p2_type)

if __name__ == "__main__":
    main(sys.argv[1:])
