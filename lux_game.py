"""
Wrapper for lux.game.Game to hold state and provide helper methods to shape
observations/actions/rewards.

Author: Jaime RuizSerra (@RuizSerra)
Date:   September 2021
"""

import numpy as np

from lux.game import Game


class LuxGame:

    def __init__(self, observation):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state.id = observation.player
            self.player_id = observation.player

    def update(self, observation: dict):
        if observation["step"] == 0:
            self.game_state._update(observation["updates"][2:])
        else:
            self.game_state._update(observation["updates"])

    def get_team_actors(self, teams=(0,)):
        self.units = []
        self.citytiles = []
        self.cities = []
        for player in [p for p in self.game_state.players if p.team in teams]:
            for unit in player.units:
                self.units.append(unit)
            for city in player.cities.values():
                for citytile in city.citytiles:
                    self.citytiles.append(citytile)
                self.cities.append(city)

        citytiles = [c for c in self.citytiles if c.team in teams]
        units = [u for u in self.units if u.team in teams]

        return {'units': units, 'citytiles': citytiles}

    def get_observation_as_tensor(self, game_state: Game):
        """Get representation of observation as HxWxC tensor

        Inspired by https://www.kaggle.com/aithammadiabdellatif/keras-lux-ai-reinforcement-learning
        """
        self.game_state = game_state
        width, height = game_state.map_width, game_state.map_height
        M = np.zeros((height, width, 4))  # map cell vector depth
        C = np.zeros((height, width, 4))  # citytile vector depth
        U = np.zeros((height, width, 6))  # unit vector depth

        for y in range(height):
            for x in range(width):
                cell = game_state.map.get_cell(x, y)
                M[y, x] = self.get_cell_as_vector(cell)
                # if cell.citytile:  # Could do this here but we need the city as well
                #     C[y, x] = self.get_citytile_as_vector(cell.citytile, city???)

        self.units = []
        self.citytiles = []
        self.cities = []
        for player in game_state.players:
            for unit in player.units:
                x, y = unit.pos.x, unit.pos.y
                # FIXME: if two units in the same citytile, one will overwrite the other in the observation
                U[y, x] = self.get_unit_as_vector(unit)
                self.units.append(unit)
            for city in player.cities.values():
                for citytile in city.citytiles:
                    x, y = citytile.pos.x, citytile.pos.y
                    C[y, x] = self.get_citytile_as_vector(citytile, city)
                    self.citytiles.append(citytile)
                self.cities.append(city)

        return np.dstack([M, U, C])

