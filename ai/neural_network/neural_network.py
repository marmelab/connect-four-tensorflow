import tensorflow as tf

from copy import deepcopy

board_width = 4
board_height = 4

# Parameters
learning_rate = 0.1

# Network Parameters
n_hidden_1 = 26
n_hidden_2 = 26
n_hidden_3 = 26
n_hidden_4 = 26
n_input = board_width * board_height * 3
n_classes = board_width

# tf Graph input
board = tf.placeholder("float", [4, 4])
x_p1 = tf.cast(tf.equal(board, -1), "float")
x_p2 = tf.cast(tf.equal(board, 1), "float")
x_empty = tf.cast(tf.equal(board, 0), "float")
x = tf.reshape(tf.concat(0,[x_p1, x_p2, x_empty]), [1, n_input])
rating = tf.placeholder("float", [4])
y = tf.reshape(rating, [1, n_classes])


def multilayer_network(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1) # tan hyperbolique ?
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.sigmoid(layer_2)
    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.sigmoid(layer_3)
    # Hidden layer with sigmoid activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.sigmoid(layer_3)
    # Output layer with softmax activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    out_layer = tf.nn.softmax(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
predict = multilayer_network(x, weights, biases)

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
