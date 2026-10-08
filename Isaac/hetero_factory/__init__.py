"""Hetero (union-scene) version of the Isaac Lab Factory tasks. See ../README.md."""

try:
    import gymnasium as gym
except ImportError:  # offline checks without gymnasium
    gym = None

if gym is not None:
    gym.register(
        id="Isaac-HeteroFactory-Direct-v0",
        entry_point=f"{__name__}.hetero_factory_env:HeteroFactoryEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hetero_factory_env_cfg:HeteroFactoryEnvCfg",
            # the official Factory PPO config, unchanged (RULES.md: no retuning)
            "rl_games_cfg_entry_point": "isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
        },
    )
    # big-list scene: each env holds only its own task's parts (hetero only)
    gym.register(
        id="Isaac-HeteroFactoryList-Direct-v0",
        entry_point=f"{__name__}.hetero_factory_list_env:HeteroFactoryListEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.hetero_factory_env_cfg:HeteroFactoryListEnvCfg",
            "rl_games_cfg_entry_point": "isaaclab_tasks.direct.factory.agents:rl_games_ppo_cfg.yaml",
        },
    )
