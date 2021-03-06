"""
Lux AI environment interface for RL-lib Multi-Agents

Authors:  Jaime Ruiz Serra (@RuizSerra)
Date:     Sep 2021
"""
import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from kaggle_environments import make

from multilux.lux_interface import LuxDefaultInterface


class LuxEnv(MultiAgentEnv):
    """
    A MultiAgentEnv only needs two methods: reset() and step().
    They both return a series of dicts where each key is an actor id (e.g. 'u_1').
    Here we use "actor" to refer to worker, cart, or citytile, to make
    the distinction with "agent", which is the overall team/player.

    All that you need to customise for your own ML engineering is the interface.
    See lux_interface.LuxDefaultInterface() for reference.

    The data flow is, at a high level:
        LuxEnv[self.env] -> LuxInterface[self.game]---(obs, rew)---> [agent]---(action)---> *repeat*

    RLlib docs: https://docs.ray.io/en/stable/rllib-package-ref.html#ray.rllib.env.MultiAgentEnv

    :param configuration: (Dict) the config dict for the LuxAI Kaggle environment
    :param debug: (Bool)
    :param interface: (LuxDefaultInterface) Defines how joint observations are
                       converted to per-actor observations, and such.
    :param agents: (Iterable) The two agents to run in the environment. Set the one training to None.
    :param train: (Bool)  Not sure, I think it needs to always be True?
    """
    def __init__(self, configuration, debug,
                 interface=LuxDefaultInterface,
                 agents=(None, "simple_agent")):
        super().__init__()

        logger.debug('Init LuxEnv')

        self._env = make("lux_ai_2021",
                         configuration=configuration, debug=debug)

        self.env = self._env.train(agents)

        self.interface_class = interface
        self.interface = None  # will be instantiated in self.reset()

        self.action_space = None
        self.observation_space = None

    def reset(self):
        """
        returns a dictionary of observations with keys being agent ids
        """
        logger.debug("==================== Environment reset ====================")
        obs = self.env.reset()
        # Instantiate interface to agent
        self.interface = self.interface_class(obs)
        obs = self.interface.observation(obs)
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
            "u_1": 1, "c_1_1": 0,
        }

        """
        # Convert actions dict to list of actions as per LuxAI spec
        actions = self.interface.actions(action_dict)
        # Apply actions to environment
        obs, reward, done, info = self.env.step(actions)
        # Convert data to dicts as per RLlib spec
        obs, reward, done, info = self.interface.ordi(obs, reward, done, info)
        return obs, reward, done, info
