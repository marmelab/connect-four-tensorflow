import tensorflow as tf
import numpy as np
from copy import deepcopy

# Parameters
learning_rate = 0.1

board_width = 7
board_height = 6

hidden_layers = [
    {
        'neurons' : 67,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : 67,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : 67,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : 67,
        # 'dropout': 0.75,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : board_width,
        'activation_function': 'softmax',
    },
]


# Network Parameters
n_input = board_width * board_height * 3

# tf Graph input
board = tf.placeholder("float", [board_width, board_height])
x_p1 = tf.cast(tf.equal(board, -1), "float")
x_p2 = tf.cast(tf.equal(board, 1), "float")
x_empty = tf.cast(tf.equal(board, 0), "float")
x = tf.reshape(tf.concat(0,[x_p1, x_p2, x_empty]), [1, n_input])
rating = tf.placeholder("float", [board_width])
y = tf.reshape(rating, [1, board_width])

def multilayer_network(x, hidden_layers):
    previous_size = n_input
    previous_layer = x
    for layer in hidden_layers:
        layer_size = layer['neurons']
        layer['weights'] = tf.Variable(tf.random_normal([previous_size, layer_size]))
        layer['biases'] = tf.Variable(tf.random_normal([layer_size]))

        layer['predict'] = tf.add(tf.matmul(previous_layer, layer['weights']), layer['biases'])

        if 'activation_function' in layer :
            if layer['activation_function'] == 'sigmoid':
                layer['predict'] = tf.sigmoid(layer['predict'])
            elif layer['activation_function'] == 'tanh':
                layer['predict'] = tf.tanh(layer['predict'])
            elif layer['activation_function'] == 'softmax':
                layer['predict'] = tf.nn.softmax(layer['predict'])

        if 'dropout' in layer:
            dropout = tf.constant(layer['dropout'])
            layer['predict'] = tf.nn.dropout(layer['predict'], dropout)

        previous_size = layer_size
        previous_layer = layer['predict']

    return hidden_layers[-1]['predict']

# Construct model
predict = multilayer_network(x, hidden_layers)

# Backward propagation

# cost = tf.reduce_mean(tf.square(y - predict))
cost = tf.contrib.losses.softmax_cross_entropy(predict, y)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y))
# cost = tf.nn.sparse_softmax_cross_entropy_with_logits(predict, y)
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predict, y))

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary('accuracy', accuracy)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


saver = tf.train.Saver()


class NeuralNetwork:

    def __init__(self, player, learn):
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        self.learn = learn
        self.player = player
        self.saved_actions = []
        try:
            saver.restore(
                self.session, 'ai/neural_network/saved_graphs/connect-four')
            print("Restored graph file")
        except ValueError:
            print("Cannot restore graph file")
        self.number_of_games = 0
        self.number_of_turns = 0
        self.correctly_guessed_turns = 0

    def __enter__(self):
        return self

    def next_move(self, game):
        predictions = self.session.run([predict], feed_dict={
            board: game.board,
        })
        legal_moves = game.get_legal_moves()
        score = None
        column = -1
        for index, prediction in enumerate(predictions[0][0]):
            if prediction > score and legal_moves[index]:
                score = prediction
                column = index

        return column

    def turn_feedback(self, game, column, opponent_would_have_played):
        self.number_of_turns += 1
        if column == opponent_would_have_played:
            self.correctly_guessed_turns += 1
        if self.learn :
            self.back_propagation(game.board, opponent_would_have_played, 1)

    def opponent_turn_feedback(self, game, column, opponent_would_have_played):
        self.number_of_turns += 1
        if column == opponent_would_have_played:
            self.correctly_guessed_turns += 1
        if self.learn :
            other_board = np.multiply(game.board, -1)
            self.back_propagation(other_board, opponent_would_have_played, 1)

    def game_feedback(self, game, status, winner):
        self.number_of_games += 1

        if self.learn and self.number_of_games % 10 == 0:
            print("Correctly guessed this batch : " + str(100 * self.correctly_guessed_turns / self.number_of_turns) + '% (' + str(self.correctly_guessed_turns) + '/' + str(self.number_of_turns) + ')')
            self.number_of_turns = 0
            self.correctly_guessed_turns = 0

            saver.save(self.session,
                       'ai/neural_network/saved_graphs/connect-four')
            print("Saving session graph")

    def back_propagation(self, game_board, column, score):
        output_data = [0 for i in range(board_width)]
        output_data[column] = score
        _, c = self.session.run([optimizer, cost], feed_dict={
            board: game_board,
            rating: output_data
        })

    def __exit__(self, type, value, traceback):
        self.session.close()
