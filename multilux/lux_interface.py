"""
Interface between LuxEnv (mostly lists, vectors) and rllib Trainer (dicts of actors).

To change the logic and do feature/reward engineering, create a new class that
inherits from this one and pass it to LuxEnv when instantiating it.

Author: Jaime Ruiz Serra (@RuizSerra)
Date:   September 2021
"""

import logging

logger = logging.getLogger(__name__)

from typing import Callable, Iterator, Union, Optional, List, Tuple
from gym import spaces
import numpy as np

from multilux.lux_game import LuxGame


class LuxDefaultInterface:

    obs_spaces = {'default': spaces.Box(low=0, high=1,
                                                 shape=(2,), dtype=np.float16)}
    act_spaces = {'default': spaces.Discrete(2)}

    def __init__(self, obs):
        # logger.debug('Init interface')
        # Instantiate game wrapper
        self.game = LuxGame(obs)
        self.game_state = self.game.update(obs)

    def ordi(self, *joint_data) -> Tuple[dict]:
        """Coordinates the conversion of each of the data types

        :param joint_data: (obs, reward, done, info) as LuxAI (kaggle env) format
        :return: (obs, reward, done, info) as dicts for RLlib trainer
        """

        funcs = [self.observation,
                 self.reward,
                 self.done,
                 self.info]

        # Update game state form env observation
        self.game_state = self.game.update(joint_data[0])
        # Get actor objects for current player from game
        actors = self.game.get_team_actors(teams=(self.game.player_id,), flat=True)

        output_data = []
        for fun, data in zip(funcs, joint_data):
            data = fun(data, actors)
            output_data.append(data)

        return tuple(output_data)

    def observation(self, joint_obs, actors=None) -> dict:
        # use self.game_state
        return {a.id: self.obs_spaces['default'].sample() for a in actors}

    def reward(self, joint_reward, actors) -> dict:
        # use self.game_state
        return {a.id: 0 for a in actors}

    def done(self, joint_done, actors) -> dict:
        # use self.game_state
        d = {a.id: True for a in actors}
        d['__all__'] = True  # turn completion
        return d

    def info(self, joint_info, actors) -> dict:
        # use self.game_state
        return {a.id: {} for a in actors}

    def actions(self, action_dict) -> list:
        """
        Takes an RLlib multi-agent style dict.
        Returns a list of LuxAI actions
        """
        # use self.game_state
        return []
