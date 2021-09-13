
# Lux AI interface to RLlib `MultiAgentsEnv`

For [Lux AI Season 1](https://www.kaggle.com/c/lux-ai-2021) Kaggle competition.

* [LuxAI](https://github.com/Lux-AI-Challenge/Lux-Design-2021)
* [RLlib-multiagents](https://docs.ray.io/en/stable/rllib-package-ref.html#ray.rllib.env.MultiAgentEnv)  
* [Kaggle environments](https://github.com/Kaggle/kaggle-environments#training)  

## TLDR
```python
import random

from ray.tune.registry import register_env
from ray.rllib.agents import ppo

from lux_env import LuxEnv

class MyEnv(LuxEnv):
    def __shape_observation(self, joint_obs, actors) -> dict:
        f = lambda o,a: [o_]  # convert joint_obs to individual_obs
        return {a: f(joint_obs, a) for a in actors}
    # (...) just need to define a couple more custom methods

register_env("lux-env", lambda x: MyEnv())

# TODO: define observation and action spaces for each actor type
u_obs_space = [] # gym.space
u_act_space = []
ct_obs_space = []
ct_act_space = []

config = {
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

trainer = ppo.PPOAgent(env="lux-env", config=config)

while True:
    print(trainer.train())
```

---
See also the [LuxPythonEnvGym](https://github.com/glmcdona/LuxPythonEnvGym) `OpenAI-gym` port by @glmcdona.

[Jaime Ruiz Serra](https://www.kaggle.com/ruizserra)
