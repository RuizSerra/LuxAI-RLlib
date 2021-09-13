
# Lux AI interface to RLlib `MultiAgentsEnv`

For [Lux AI Season 1](https://www.kaggle.com/c/lux-ai-2021) Kaggle competition.

* [LuxAI](https://github.com/Lux-AI-Challenge/Lux-Design-2021)
* [RLlib-multiagents](https://docs.ray.io/en/stable/rllib-package-ref.html#ray.rllib.env.MultiAgentEnv)  
* [Kaggle environments](https://github.com/Kaggle/kaggle-environments#training)  

## TLDR
```python
from ray.tune.registry import register_env
from lux_env import LuxEnv

# (1) Define your custom interface for (obs, reward, done, info, actions) ---
from lux_interface import LuxDefaultInterface

class MyInterface(LuxDefaultInterface):
    def observation(self, joint_obs, actors) -> dict:
        return {a: [0] for a in actors}

    def reward(self, joint_reward, actors) -> dict:
        return {a: 0 for a in actors}

    def done(self, joint_done, actors) -> dict:
        return {a: True for a in actors}

    def info(self, joint_info, actors) -> dict:
        return {a: {} for a in actors}

    def actions(self, action_dict) -> list:
        return []

f = lambda x: LuxEnv(configuration, debug,
                     interface=MyInterface,
                     agents=(None, "simple_agent"),
                     train=True)
register_env("lux-env", f)

# (2) Define observation and action spaces for each actor type --------------
u_obs_space = [] # TODO: gym.space
u_act_space = []
ct_obs_space = []
ct_act_space = []

# (3) Instantiate agent ------------------------------------------------------
import random
from ray.rllib.agents.ppo import ppo

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

trainer = ppo.PPOTrainer(env="lux-env", config=config)


# (4) Train away -------------------------------------------------------------
while True:
    print(trainer.train())
```

---
See also the [LuxPythonEnvGym](https://github.com/glmcdona/LuxPythonEnvGym) `OpenAI-gym` port by @glmcdona.

[Jaime Ruiz Serra](https://www.kaggle.com/ruizserra)
