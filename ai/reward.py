import tensorflow as tf

from bitmasks import bitmasks

board = tf.placeholder('float', [4, 4])
masks = tf.constant(bitmasks, 'bool', [10, 4, 4])

reward_positive = lambda: tf.constant(1)
reward_negative = lambda: tf.constant(-1)
reward_none = lambda: tf.constant(0)

# put board and mask in 1d
flattened_masks = tf.reshape(masks, [10 * 4 * 4])
flattened_boards = tf.reshape(tf.tile(board, [10, 1]), [10 * 4 * 4])

# use a boolean mask to extract a particular chunk
extract_chunks = tf.boolean_mask(flattened_boards, flattened_masks)

# go back to 2d
reconstruct_chunks = tf.reshape(extract_chunks, [10, 4])

# which of the chunks have 4 consecutive 1s ?
consecutive_chunks = tf.reduce_prod(reconstruct_chunks, 1)

# game is lost if any of the chunks sum is 16 (2 * 2 * 2 * 2)
opponent_has_consecutive_chunks = tf.equal(consecutive_chunks, 16)
is_game_lost = tf.reduce_any(opponent_has_consecutive_chunks)

# game is lost if any of the chunks sum is 1 (1 * 1 * 1 * 1)
player_has_consecutive_chunks = tf.equal(consecutive_chunks, 1)
is_game_won = tf.reduce_any(player_has_consecutive_chunks)

get_board_reward = tf.case(
    {is_game_won: reward_positive, is_game_lost: reward_negative}, default=reward_none, exclusive=True)
