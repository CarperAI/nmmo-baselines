from argparse import Namespace

import nmmo
import pufferlib
import pufferlib.emulation
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper
from syllabus.core import PettingZooSyncWrapper as SyllabusSyncWrapper

from syllabus_wrapper import SyllabusSeedWrapper


class Config(
    nmmo.core.config.Medium,
    nmmo.core.config.Terrain,
    nmmo.core.config.Resource,
    nmmo.core.config.Combat,
    nmmo.core.config.NPC,
    nmmo.core.config.Progression,
    nmmo.core.config.Item,
    nmmo.core.config.Equipment,
    nmmo.core.config.Profession,
    nmmo.core.config.Exchange,
):
    """Configuration for Neural MMO."""

    def __init__(self, env_args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("MAP_FORCE_GENERATION", env_args.map_force_generation)
        self.set("PLAYER_N", env_args.num_agents)
        self.set("HORIZON", env_args.max_episode_length)
        self.set("MAP_N", env_args.num_maps)
        self.set(
            "PLAYER_DEATH_FOG",
            env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None,
        )
        self.set("PATH_MAPS", f"{env_args.maps_path}/{env_args.map_size}/")
        self.set("MAP_CENTER", env_args.map_size)
        self.set("NPC_N", env_args.num_npcs)
        self.set("TASK_EMBED_DIM", env_args.task_size)
        self.set("RESOURCE_RESILIENT_POPULATION", env_args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", env_args.spawn_immunity)

        self.set("GAME_PACKS", [(nmmo.core.game_api.AgentTraining, 1)])
        self.set("CURRICULUM_FILE_PATH", env_args.curriculum_file_path)


def make_env_creator(
    reward_wrapper_cls: BaseParallelWrapper, syllabus_wrapper=False, syllabus=None
):
    def env_creator(*args, **kwargs):
        """Create an environment."""
        env = nmmo.Env(Config(kwargs["env"]))  # args.env is provided as kwargs
        env = reward_wrapper_cls(env, **kwargs["reward_wrapper"])

        # Add Syllabus task wrapper
        if syllabus_wrapper or syllabus is not None:
            env = SyllabusSeedWrapper(env)

        # Use syllabus curriculum if provided
        if syllabus is not None:
            env = SyllabusSyncWrapper(
                env,
                env.task_space,
                syllabus.components,
                batch_size=8,
            )

        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator
