# Snake Environment

## Overview

This environment recreates the classic game of Snake. The snake is made of connected tiles: a head, body segments, and a tail.

At every step, the agent chooses one of four directions:

- Up
- Down
- Left
- Right

The snake moves one tile per step, and the rest of the body follows the head. When the snake eats a fruit, it grows by one tile.

The goal is to grow as large as possible by collecting fruits while avoiding collisions.

## Board

The environment uses a 10×10 grid.

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
