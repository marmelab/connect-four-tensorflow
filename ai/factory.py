from minimax.minimax import Minimax
from neural_network.neural_network import NeuralNetwork
from human.human import Human


def create(ai_name, player, **kwargs):
    if ai_name == 'minimax':
        return Minimax(player, kwargs['level'])
    elif ai_name == 'neural_network':
        return NeuralNetwork(player, kwargs['learn'])
    elif ai_name == 'human':
        return Human(player)
    else:
        raise Exception('No AI named ' + ai_name)
