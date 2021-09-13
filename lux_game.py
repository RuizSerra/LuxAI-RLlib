"""
Wrapper for lux.game.Game to hold state and provide helper methods to shape
observations/actions/rewards.

Author: Jaime RuizSerra (@RuizSerra)
Date:   September 2021
"""

from lux.game import Game


class LuxGame:

    def __init__(self, observation):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state.id = observation.player
            self.team = observation.player

    def update(self, observation: dict):
        if observation["step"] == 0:
            self.game_state._update(observation["updates"][2:])
        else:
            self.game_state._update(observation["updates"])

    def get_team_actors(self):

        citytiles = [c for c in self.citytiles if c.team == self.team]

