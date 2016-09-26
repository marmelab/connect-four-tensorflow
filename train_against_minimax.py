import tensorflow as tf

from ai.reward import get_board_reward, play
from ai.bitmasks import bitmasks
from random import randint, random
from ai.minimax import Minimax


def minimax(state):
    m = Minimax(state)
    best_move, value = m.bestMove(7, state, 1)
    return best_move


def sigmaprime(x):
    return tf.mul(tf.sigmoid(x), tf.sub(tf.constant(1.0), tf.sigmoid(x)))

ai_player = 1
opponent = -1
board_width = 4
board_height = 4


# currentPlayer = tf.Variable(1)


reward_impossible = lambda: tf.constant(-1.1, 'float')
batch_size = 100

inputs = tf.Variable(tf.zeros_initializer([16]))

switch_turns = inputs.assign(tf.mul(inputs, -1))


two_d_board = tf.reshape(inputs, [4, 4])

reinit_inputs = inputs.assign(tf.zeros_initializer([16]))
# reinit_inputs = inputs.assign([
#     0, 0, 0, 0,
#     0, 0, 0, -1,
#     1, -1, 0, -1,
#     1, -1, 1, -1
# ])

output = tf.placeholder('float')

middle = 30

hidden_layer_1 = {
    'weights': tf.Variable(tf.random_uniform([board_height * board_width, middle])),
    'biases': tf.Variable(tf.random_uniform([1, middle])),
}


hidden_layer_2 = {
    'weights': tf.Variable(tf.random_uniform([middle, board_width])),
    'biases': tf.Variable(tf.random_uniform([1, board_width]))
}


saver = tf.train.Saver()

# forward feed
z1 = tf.add(tf.matmul([inputs], hidden_layer_1[
            'weights']), hidden_layer_1['biases'])
a1 = tf.sigmoid(z1)
z2 = tf.add(tf.matmul(z1, hidden_layer_2['weights']), hidden_layer_2['biases'])
a2 = tf.sigmoid(z2)

# play
# get column from board
column_index = tf.to_int32(output)
column = tf.slice(tf.transpose(tf.reshape(inputs, [4, 4])), [
                  column_index[0], 0], [1, 4])
# check if empty
boolean_column = tf.equal(column, 0)
# get lower empty row
empty_rows = tf.where(tf.reshape(boolean_column, [4]))
lower_row = tf.argmax(tf.reshape(empty_rows, [-1]), 0)
# update board
row_index = tf.to_int32(lower_row[0])
play = tf.scatter_update(inputs, row_index * 4 + column_index[
    0], ai_player)
is_move_possible = tf.not_equal(tf.shape(empty_rows)[0], 0)

# test_board_full = tf.shape(tf.where(tf.equal(inputs, 0.)))[0]
is_board_full = tf.equal(tf.shape(tf.where(tf.equal(inputs, 0.)))[0], 0)

# accuracy of move
get_board_reward_callable = lambda: get_board_reward(inputs)
diff = tf.cond(is_move_possible, get_board_reward_callable, reward_impossible)

# backward propagation
d_z2 = tf.mul(diff, sigmaprime(z2))
d_b2 = d_z2
d_w2 = tf.matmul(tf.transpose(a1), d_z2)

d_a1 = tf.matmul(d_z2, tf.transpose(hidden_layer_2['weights']))
d_z1 = tf.mul(d_a1, sigmaprime(z1))
d_b1 = d_z1
d_w1 = tf.matmul(tf.transpose([inputs]), d_z1)

eta = tf.constant(0.5)
step = [
    tf.assign(
        hidden_layer_1['weights'],
        tf.sub(hidden_layer_1['weights'], tf.mul(eta, d_w1))
    ),
    tf.assign(
        hidden_layer_1['biases'],
        tf.sub(
            hidden_layer_1['biases'],
            tf.mul(
                eta,
                tf.reduce_mean(d_b1, reduction_indices=[0])
            )
        )
    ),
    tf.assign(
        hidden_layer_2['weights'],
        tf.sub(hidden_layer_2['weights'], tf.mul(eta, d_w2))
    ),
    tf.assign(
        hidden_layer_2['biases'],
        tf.sub(
            hidden_layer_2['biases'],
            tf.mul(
                eta,
                tf.reduce_mean(d_b2, reduction_indices=[0])
            )
        )
    )
]


best_column = tf.argmax(a2, 1)

pretty_print = tf.reshape(inputs, [4, 4])

randomness = 0.25

number_of_epochs = 200000
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    saver.restore(session, "connect-four")
    game_rules_broken = 0
    won_games_p1 = 0
    won_games_p2 = 0
    nb_draw = 0
    for epoch in range(number_of_epochs):
        turn = 0
        session.run([reinit_inputs])
        while turn < board_width * board_height:

            # Predict action to play
            action = session.run([best_column])

            if turn % 2 == 0:
                # play against minimax
                game_board = session.run([two_d_board])
                action[0] = minimax(game_board[0])
                # print "minimax choice : " + str(action)
            else:
                # print "neural network choice : " + str(action)
                # Add some randomness to the neuronal network plays
                if random() < randomness:
                    action[0] = randint(0, board_width - 1)

            # Check if predicted move is possible
            current_move_possible = session.run(
                [is_move_possible], feed_dict={output: action[0]})

            board_full = session.run(
                [is_board_full], feed_dict={output: action[0]})

            if current_move_possible[0] and not board_full[0]:
                board = session.run([play], feed_dict={output: action[0]})

            # Rate it
            board_rating = session.run(
                [diff], feed_dict={output: action[0], is_move_possible: current_move_possible[0]})

            if board_rating[0] != 0.:
                if abs(board_rating[0] - -1.1) < 1e-6:
                    game_rules_broken += 1

                elif board_rating[0] == 1 and turn % 2 == 0:
                    won_games_p1 += 1s
                elif board_rating[0] == 1 and turn % 2 == 1:
                    won_games_p2 += 1

                # Do back propagation if there's something to rate
                weights = session.run(
                    [step], feed_dict={
                        output: action[0],
                        is_move_possible: current_move_possible[0],
                        diff: board_rating[0]
                    }
                )
                break
            elif board_full[0]:
                nb_draw += 1
                break
            session.run([switch_turns])
            turn += 1
        if epoch % 2000 == 0:
            saver.save(session, "connect-four")
            print "Saved network"
        if epoch % 100 == 0:
            print session.run([pretty_print])
            print "broken rules : " + str(game_rules_broken) + " / " + str(epoch + 1)
            print "won games p1 (minimax): " + str(won_games_p1) + " / " + str(epoch + 1)
            print "won games p2 (neural network): " + str(won_games_p2) + " / " + str(epoch + 1)
            print "draw : " + str(nb_draw) + " / " + str(epoch + 1)
