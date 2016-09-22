import tensorflow as tf

from ai.bitmasks import *
from ai.reward import *

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    game_board = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    print session.run(is_aligned(), feed_dict={
        board: game_board,
        mask: bitmasks[2]
    })
