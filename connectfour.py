import tensorflow as tf

from bitmasks import *

board = tf.placeholder('float', [4, 4])
mask = tf.placeholder('bool', [4, 4])

extract_chunk = tf.boolean_mask(board, mask)

is_aligned = tf.equal(tf.reduce_prod(extract_chunk), 1)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())

    game_board = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    print session.run(is_aligned, feed_dict={
        board: game_board,
        mask: bitmasks[3]
    })
