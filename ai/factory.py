from minimax.minimax import Minimax
from neural_network.neural_network import NeuralNetwork
from human.human import Human
from kerasai.kerasai import KerasAi


def create(ai_type, player):
    tab_ai = ai_type.split(':')
    ai_name = tab_ai[0]
    if ai_name == 'minimax':
        return Minimax(player, tab_ai[1])
    elif ai_name == 'neural_network':
        return NeuralNetwork(player, tab_ai[1])
    elif ai_name == 'keras':
        return KerasAi(player, tab_ai[1])
    elif ai_name == 'human':
        return Human(player)
    else:
        raise Exception('No AI named ' + ai_name)
