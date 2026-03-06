# Board dimensions
BOARD_SIZE: int = 4

# Action indices
ACTION_UP: int = 0
ACTION_DOWN: int = 1
ACTION_LEFT: int = 2
ACTION_RIGHT: int = 3

ACTION_NAMES: dict[int, str] = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}

# Tile spawn probabilities
TILE_2_PROB: float = 0.9  # probability of spawning a 2 tile
TILE_4_PROB: float = 0.1  # probability of spawning a 4 tile

# Win condition
WIN_TILE: int = 2048
