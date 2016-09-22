import tensorflow as tf

from ai.reward import board, get_board_reward


with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    game_board = [
        [0, 2, 0, 0],
        [1, 2, 1, 1],
        [0, 2, 0, 0],
        [0, 2, 0, 0],
    ]

    print session.run(get_board_reward, feed_dict={
        board: game_board
    })
