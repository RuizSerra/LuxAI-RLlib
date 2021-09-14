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
            self.player_id = observation.player

    def update(self, observation: dict):
        if observation["step"] == 0:
            self.game_state._update(observation["updates"][2:])
        else:
            self.game_state._update(observation["updates"])

    def get_state(self):
        return self.game_state

    def get_team_actors(self, teams=(0,), flat=False):
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

        citytiles = [c for c in self.citytiles if c.team in teams]  # TODO: citytiles need ids
        units = [u for u in self.units if u.team in teams]

        if flat:
            return units + citytiles

        return {'units': units, 'citytiles': citytiles}
