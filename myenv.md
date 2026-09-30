# Snake Environment

## Overview

This environment recreates the classic game of Snake. The snake is made of connected tiles: a head, body segments, and a tail.

The snake moves one tile per step, and the rest of the body follows the head. When the snake eats a fruit, it grows by one tile.

The goal is to grow as large as possible by collecting fruits while avoiding collisions.

## Board

The environment uses a 10×10 grid.  Each tile in this grid can be in one of 5 states:  
- Blank
- Fruit
- Head
- Body
- Tail

Example:

```text
..........
....@.....
...TBH....
..........
..........
..........
..........
..........
..........
..........
```

As such, the observation space is a Box with a shape of (100,100) and a value range from 0 to 4.

## Actions

At every step, the agent chooses one of four directions:

- Up
- Down
- Left
- Right

As such, the action space is a Discrete space with 4 elements.  

## Reward Structure

Reward points are earned or removed in 4 conditions.  

- The agent "completes" the game (fills up all the squares).  [+10]
- The agent "consumes" a fruit. [+1]
- The agent collides with a wall. [-10]
- The agent collides with its own body.  [-10]

## Termination

The game terminates in three conditions.
- The agent "completes" the game (fills up all the squares). 
- The agent collides with a wall. 
- The agent collides with its own body.  

## Transition Probability

Three factors are randomized.  
- The agent's starting location, randomized to (N-1) * (M-1) squares.
- The number of fruits spawned on each wave.  From 1 to half the snake's length floored.
- The location of each fruit's spawn.  Randomized to (N) * (M) squares. 

## __init__

No special parameters are needed for this environment.  Only a render mode specified as a string.