import tensorflow as tf

from copy import deepcopy

board_width = 4
board_height = 4

# Parameters
learning_rate = 0.1

# Network Parameters
n_hidden_1 = 10
n_hidden_2 = 10
n_input = board_width * board_height
n_classes = board_width

# tf Graph input
board = tf.placeholder("float", [4, 4])
x = tf.reshape(board, [1, n_input])
rating = tf.placeholder("float", [4])
y = tf.reshape(rating, [1, n_classes])


def multilayer_network(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
predict = multilayer_network(x, weights, biases)

# Backward propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
                rating = 1

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
