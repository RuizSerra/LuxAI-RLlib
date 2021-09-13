"""
Lux AI environment interface for RL-lib Multi-Agents

Authors:  Jaime Ruiz Serra (@RuizSerra)
Date:     Sep 2021
"""
import logging
from typing import Callable, Optional, Tuple

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID

from kaggle_environments import make

# logger = logging.getLogger(__name__)


class LuxEnv(MultiAgentEnv):
    """
    A MultiAgentEnv only needs two methods: reset() and step().
    They both return a series of dicts where each key is an actor id (e.g. 'u_1').
    Here we use "actor" to refer to worker, cart, or citytile, to make
    the distinction with "agent", which is the overall team/player.

    The data flow is, at a high level:
        [self.env -> self.game -> self.shape_stuff]---(obs, rew)---> [agent]---(action)---> *repeat*

    RLlib docs: https://docs.ray.io/en/stable/rllib-package-ref.html#ray.rllib.env.MultiAgentEnv
    """
    def __init__(self, configuration, debug, game_state: Game, train=False):
        super().__init__()

        self.env = make("lux_ai_2021",
                        configuration=configuration, debug=debug)
        if train:  # ???
            self.env = self.env.train(agents)

        self.action_space = None
        self.observation_space = None

        self.game = None  # will be set to LuxGame(obs) in self.reset()

    def reset(self):
        """
        returns a dictionary of observations with keys being agent ids
        """
        obs = self.env.reset()

        self.game = LuxGame(obs)
        self.game.update(obs)

        obs = self.__shape_observation(obs)

        return obs

    def step(self, action_dict):
        """
        Takes as input a dictionary of actions with keys being agent ids
        Must return a dictionary of observations, rewards, dones (boolean) and info.
        Again, the keys for all these dictionaries are agent ids.

        dones dictionary has an additional key '__all__'
        which must be True only when all agents have completed the episode

        The action_dict always contains actions for observations returned in the previous timestep

        :param action_dict:

        action_dict={
            "car_0": 1, "car_1": 0, "traffic_light_1": 2,
        }


        :return:
        """
        obs, reward, done, info = self.env.step(action_dict)

        self.game.update(obs)

        obs = self.__shape_observation(obs)
        reward = self.__shape_reward(reward)
        done = self.__shape_dones(done)
        info = self.__shape_info(info)

        return obs, reward, done, info

    def __shape_observation(self, obs) -> dict:
        """
        {
            "car_0": [2.4, 1.6],
            "car_1": [3.4, -3.2],
            "traffic_light_1": [0, 3, 5, 1],
        }
        """
        return {}

    def __shape_reward(self, reward) -> dict:
        """{
            "car_0": 3,
            "car_1": -1,
            "traffic_light_1": 0,
        }"""
        return {}

    def __shape_dones(self, dones) -> dict:
        """
        {
            "car_0": False,    # car_0 is still running
            "car_1": True,     # car_1 is done
            "__all__": False,  # the env is not done
        }
        :param dones:
        :return:
        """
        return {}

    def __shape_info(self, info) -> dict:
        """
        {
            "car_0": {},  # info for car_0
            "car_1": {},  # info for car_1
        }
        :param info:
        :return:
        """
        return {}


