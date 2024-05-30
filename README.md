# Game-of-Life
An implementation of the Game of Life by John Conway (yet another copy of Game of Life).

The goal of this repository is to try to fasten the calculations from a state to another, and also to play with the rules.

Two different approaches have been used for now to accelerate:
- Concurrent processing using the `concurrent` Python package
- Neural networks with `pytorch`

The neural networks didn't go through a learning process as we can just put the weights we want so that it complies with the Game of Life (or any variant) and also facilitate changes in board size.

The torch version uses `mps` device to use the Apple silicon GPU. You might want to use `cuda` instead if you have a Nvidia GPU. 

**Warning** 
The version of `pytorch` described in `requirements.txt` is a nightly build (as of May 30st 2024) in order to use `mps` device. You might want to use another build if you don't have an apple chip.
