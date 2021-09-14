"""
Interface between LuxEnv (mostly lists, vectors) and rllib Trainer (dicts of actors).

To change the logic and do feature/reward engineering, create a new class that
inherits from this one and pass it to LuxEnv when instantiating it.

Author: Jaime Ruiz Serra (@RuizSerra)
Date:   September 2021
"""

from typing import Callable, Iterator, Union, Optional, List, Tuple
from gym import spaces
import numpy as np


class LuxDefaultInterface:
    """
    Note: any operations with self.game should be read only ideally
    """

    def __init__(self, game):
        self.game = game

    def ordi(self, *args) -> Tuple[dict]:
        funcs = [self.observation,
                 self.reward,
                 self.done,
                 self.info]

        actors = args.pop()

        joint_data = args
        output_data = []
        for fun, data in zip(funcs, joint_data):
            data = fun(data, actors)
            output_data.append(data)

        return tuple(output_data)

    def observation(self, joint_obs, actors) -> dict:
        return {a: np.array([0, 0]) for a in actors}

    def reward(self, joint_reward, actors) -> dict:
        return {a: 0 for a in actors}

    def done(self, joint_done, actors) -> dict:
        d = {a: True for a in actors}
        d['__all__'] = True  # turn completion
        return d

    def info(self, joint_info, actors) -> dict:
        return {a: {} for a in actors}

    def actions(self, action_dict) -> list:
        return []

