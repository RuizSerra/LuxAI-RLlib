
# Lux AI interface to RLlib `MultiAgentsEnv`

For [Lux AI Season 1](https://www.kaggle.com/c/lux-ai-2021) Kaggle competition.

* [LuxAI repo](https://github.com/Lux-AI-Challenge/Lux-Design-2021)
* [RLlib-multiagents docs](https://docs.ray.io/en/stable/rllib-package-ref.html#ray.rllib.env.MultiAgentEnv)  
* [Kaggle environments repo](https://github.com/Kaggle/kaggle-environments#training)

Please let me know if you use this, I'd like to see what people build with it!

## TL;DR

The only thing you need to customise is the interface class (inheriting from 
`multilux.lux_interface.LuxDefaultInterface`). The interface needs to:
* Implement four "toward-agent" methods:
    - `observation(joint_observation, actors)`
    - `reward(joint_reward, actors)`
    - `done(joint_done, actors)`
    - `info(joint_info, actors)`
* Implement one "toward-environment" method:    
    - `actions(action_dict)`
* Manage its own `actor id` creation, assignment, etc. 
  (hint citytiles don't have ids in the game engine)

### Implementation diagram

![Diagram](img/img.png)

### Example for training

```python
import numpy as np

# (1) Define your custom interface for (obs, reward, done, info, actions) ---
from multilux.lux_interface import LuxDefaultInterface

class MyInterface(LuxDefaultInterface):
    def observation(self, joint_obs, actors) -> dict:
        return {a: np.array([0, 0]) for a in actors}

    def reward(self, joint_reward, actors) -> dict:
        return {a: 0 for a in actors}

    def done(self, joint_done, actors) -> dict:
        return {a: True for a in actors}

    def info(self, joint_info, actors) -> dict:
        return {a: {} for a in actors}

    def actions(self, action_dict) -> list:
        return []
    
# (2) Register environment --------------------------------------------------
from ray.tune.registry import register_env
from multilux.lux_env import LuxEnv


def env_creator(env_config):
    
    configuration = env_config.get(configuration, {})
    debug = env_config.get(debug, False)
    interface = env_config.get(interface, MyInterface)
    agents = env_config.get(agents, (None, "simple_agent"))
    
    return LuxEnv(configuration, debug,
                     interface=interface,
                     agents=agents,
                     train=True)

register_env("multilux", env_creator)

# (3) Define observation and action spaces for each actor type --------------
from gym import spaces

u_obs_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float16)
u_act_space = spaces.Discrete(2)
ct_obs_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float16)
ct_act_space = spaces.Discrete(2)

# (4) Instantiate agent ------------------------------------------------------
import random
from ray.rllib.agents import ppo

config = {
    "env_config": {},
    "multiagent": {
        "policies": {
            # the first tuple value is None -> uses default policy
            "unit-1": (None, u_obs_space, u_act_space, {"gamma": 0.85}),
            "unit-2": (None, u_obs_space, u_act_space, {"gamma": 0.99}),
            "citytile": (None, ct_obs_space, ct_act_space, {}),
        },
        "policy_mapping_fn":
            lambda agent_id:
                "citytile"  # Citytiles always have the same policy
                if agent_id.startswith("u_")
                else random.choice(["unit-1", "unit-2"])  # Randomly choose from unit policies
    },
}

trainer = ppo.PPOTrainer(env=LuxEnv, config=config)

# (5) Train away -------------------------------------------------------------
while True:
    print(trainer.train())
```

See [`examples/training.py`](examples/training.py)

---
See also the [LuxPythonEnvGym](https://github.com/glmcdona/LuxPythonEnvGym) `OpenAI-gym` port by @glmcdona.

[Jaime Ruiz Serra](https://www.kaggle.com/ruizserra)
