from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model

import numpy as np


board_width = 7
board_height = 6
n_input = board_width * board_height

class KerasAi:

    def __init__(self, player, learn):
        self.model = Sequential()

        self.model.add(Dense(output_dim=42, input_dim=n_input))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(output_dim=42, input_dim=42))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(output_dim=42, input_dim=42))
        self.model.add(Activation("sigmoid"))
        self.model.add(Dense(output_dim=board_width))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        try:
            self.model = load_model('connect-four.h5')
            print("Restored graph file")
        except IOError:
            print("Cannot restore graph file")

        self.x_train = []
        self.y_train = []

        self.learn = learn

        self.number_of_games = 0
        self.number_of_turns = 0
        self.correctly_guessed_turns = 0

    def __enter__(self):
        return self

    def next_move(self, game):
        predictions = self.model.predict_proba(np.reshape(game.board,[1, n_input]), batch_size=32, verbose=0)

        legal_moves = game.get_legal_moves()
        score = None
        column = -1
        for index, prediction in enumerate(predictions[0]):
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

    def back_propagation(self, game_board, column, score):
        output_data = [0 for i in range(board_width)]
        output_data[column] = score

        self.x_train.append(np.reshape(game_board,[n_input]))
        self.y_train.append(np.reshape(output_data,[board_width]))

        # self.model.fit(np.reshape(game_board,[1, 16]), np.reshape(output_data,[1, 4]), nb_epoch=1, batch_size=1)
        if self.learn and self.number_of_turns % 100 == 0:
            print("Correctly guessed this batch : " + str(100 * self.correctly_guessed_turns / self.number_of_turns) + '% (' + str(self.correctly_guessed_turns) + '/' + str(self.number_of_turns) + ')')

            print("Starting training")

            self.model.fit(np.asarray(self.x_train), np.asarray(self.y_train), nb_epoch=500, batch_size=len(self.x_train))

            self.x_train = []
            self.y_train = []
            self.number_of_turns = 0
            self.correctly_guessed_turns = 0

            print("Saving session graph")
            self.model.save('connect-four.h5')


    def game_feedback(self, game, status, winner):
        self.number_of_games += 1


    def __exit__(self, type, value, traceback):
        pass
