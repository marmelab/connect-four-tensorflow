import tensorflow as tf

from bitmasks import bitmasks

masks = tf.constant(bitmasks, 'bool', [10, 4, 4])
ai_player = tf.constant(1, 'float')


def play(inputs, output):
    # get column from board
    column_index = tf.to_int32(output)
    column = tf.slice(tf.reshape(inputs, [4, 4]), [column_index[0], 0], [1, 4])
    # check if empty
    boolean_column = tf.equal(column, 0)
    # get lower empty row
    empty_rows = tf.where(boolean_column)
    lower_row = tf.argmax(empty_rows, 1)
    # update board
    row_index = tf.to_int32(lower_row[0])
    return tf.scatter_update(inputs, column_index[0] * 4 + row_index, ai_player)


def get_board_reward(board):
    board = tf.reshape(board, [4, 4])

    # board = tf.placeholder('float', [4, 4])
    reward_positive = lambda: tf.constant(1, 'float')
    reward_negative = lambda: tf.constant(-1, 'float')
    reward_none = lambda: tf.constant(0, 'float')

    # put board and mask in 1d
    flattened_masks = tf.reshape(masks, [10 * 4 * 4])
    flattened_boards = tf.reshape(tf.tile(board, [10, 1]), [10 * 4 * 4])

    # use a boolean mask to extract a particular chunk
    extract_chunks = tf.boolean_mask(flattened_boards, flattened_masks)

    # go back to 2d
    reconstruct_chunks = tf.reshape(extract_chunks, [10, 4])

    # which of the chunks have 4 consecutive 1s ?
    consecutive_chunks = tf.reduce_sum(reconstruct_chunks, 1)

    # game is lost if any of the chunks sum is -4 (-1 + -1 + -1 + -1)
    opponent_has_consecutive_chunks = tf.equal(consecutive_chunks, -4)
    is_game_lost = tf.reduce_any(opponent_has_consecutive_chunks)

    # game is lost if any of the chunks sum is 4 (1 + 1 + 1 + 1)
    player_has_consecutive_chunks = tf.equal(consecutive_chunks, 4)
    is_game_won = tf.reduce_any(player_has_consecutive_chunks)

    return tf.case(
        {is_game_won: reward_positive, is_game_lost: reward_negative}, default=reward_none, exclusive=True)
