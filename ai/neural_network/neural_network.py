import tensorflow as tf

from copy import deepcopy

# Parameters
learning_rate = 0.1

board_width = 4
board_height = 4

hidden_layers = [
    {
        'neurons' : 26,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : 26,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : 26,
        'activation_function': 'sigmoid',
    },
    {
        'neurons' : 26,
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
rating = tf.placeholder("float", [4])
y = tf.reshape(rating, [1, board_width])

def multilayer_network(x, hidden_layers):
    previous_size = n_input
    previous_layer = x
    for layer in hidden_layers:
        layer_size = layer['neurons']
        layer['weights'] = tf.Variable(tf.random_normal([previous_size, layer_size]))
        layer['biases'] = tf.Variable(tf.random_normal([layer_size]))

        layer['predict'] = tf.add(tf.matmul(previous_layer, layer['weights']), layer['biases'])

        if layer['activation_function'] == 'sigmoid':
            layer['predict'] = tf.sigmoid(layer['predict'])
        elif layer['activation_function'] == 'softmax':
            layer['predict'] = tf.nn.softmax(layer['predict'])

        previous_size = layer_size
        previous_layer = layer['predict']

    return hidden_layers[-1]['predict']

# Construct model
predict = multilayer_network(x, hidden_layers)

# Backward propagation

# cost = tf.reduce_mean(tf.square(y - predict))
cost = tf.contrib.losses.softmax_cross_entropy(predict, y)
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predict, y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)


saver = tf.train.Saver()


class NeuralNetwork:

    def __init__(self, player):
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())
        self.player = player
        self.saved_actions = []
        try:
            saver.restore(
                self.session, 'ai/neural_network/saved_graphs/connect-four')
            print("Restored graph file")
        except ValueError:
            print("Cannot restore graph file")
        self.number_of_games = 0

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

        self.saved_actions.append({
            'board': deepcopy(game.board),
            'column': column,
        })

        return column

    def turn_feedback(self, player, column):
        pass

    def game_feedback(self, game, status, winner):
        self.number_of_games += 1

        if winner != 0:
            if winner == self.player:
                for action in self.saved_actions:
                    self.back_propagation(
                        action['board'], action['column'], 1)

        self.saved_actions = []

        if self.number_of_games % 1000 == 0:
            saver.save(self.session,
                       'ai/neural_network/saved_graphs/connect-four')
            print("Saving session graph")

    def back_propagation(self, game_board, column, score):
        output_data = [0., 0., 0., 0.]
        output_data[column] = score

        _, c = self.session.run([optimizer, cost], feed_dict={
            board: game_board,
            rating: output_data
        })

    def __exit__(self, type, value, traceback):
        self.session.close()
