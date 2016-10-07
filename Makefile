.PHONY: run test help
.DEFAULT_GOAL := help

help:
	@grep -P '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Development ==================================================================

run:
	python train.py --p1 minimax --p2 neural_network --iterations 200

# Tests ========================================================================

test:
	python -m unittest discover
