"""Entry point: launch a human-playable game of 2048."""

from env.game2048_env import Game2048Env
from ui.pygame_ui import PygameUI

env = Game2048Env()
ui = PygameUI(env)
ui.run()
