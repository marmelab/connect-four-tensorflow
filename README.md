# connect-four-tensorflow

Connect four solver using neural networks and tensorflow

## Run

Runs the script

```
python train.py --p1 <xxx> --p2 <xxx> --board_width <x> --board_height <x> --iterations <x> --randomness <x> -v
```

- `p1` : which algorithm the first player should use (human, minimax, neural_network)
- `p2` : which algorithm the second player should use (human, minimax, neural_network)
- `board_width` : width of the board (default: 4)
- `board_height` : height of the board (default: 4)
- `iterations` : number of games played (default: 1)
- `randomness` : randomness of first player (default: 0.25)
- `v`: verbose output

Example :
```
python train.py --p1 minimax --p2 neural_network --iterations 200000 --randomness 0.1
```

## Test

Runs all the tests of the project

```
make test
```

## Contribute

Where the things happen :

- train.py : handles the differnt games and calls the different AIs
- ai/neural-network/neural-network.py : create the neural network, uses it for prediction and handles back propagation
- connectfour/game.py : handles connect four game, makes AIs play and check who wins
