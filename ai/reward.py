import tensorflow as tf

board = tf.placeholder('float', [4, 4])
mask = tf.placeholder('bool', [4, 4])


def is_aligned():
    # use a boolean mask to extract a particular chunk
    extract_chunk = tf.boolean_mask(board, mask)
    # use the product of all cells to check if it's equal to 1
    # which means it's only 1s
    return tf.equal(tf.reduce_prod(extract_chunk), 1)
