from argparse import Namespace
from collections import defaultdict
import math

import nmmo
import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor

class Config(nmmo.config.Default):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.MAP_FORCE_GENERATION = False
        self.PLAYER_N = args.num_agents
        self.HORIZON = args.max_episode_length
        self.MAP_N = args.num_maps
        self.PLAYER_DEATH_FOG = args.death_fog_tick
        self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
        self.MAP_CENTER = args.map_size
        self.NPC_N = args.num_npcs
        self.CURRICULUM_FILE_PATH = args.tasks_path
        self.TASK_EMBED_DIM = args.task_size

        self.COMMUNICATION_SYSTEM_ENABLED = False

        self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity

class Postprocessor(StatPostprocessor):
    def __init__(self, env, teams, team_id,
      replay_save_dir=None,
      sqrt_achievement_rewards=False,
      heal_bonus_weight=0,
      explore_bonus_weight=0,
      clip_unique_event=3,
    ):
        super().__init__(env, teams, team_id, replay_save_dir)
        self.sqrt_achievement_rewards = sqrt_achievement_rewards
        self.heal_bonus_weight = heal_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

    def reset(self, team_obs, dummy=False):
        super().reset(team_obs, dummy)

    def actions(self, team_actions, step):
        """Called in _prestep() via _handle_actions().
           See https://github.com/PufferAI/PufferLib/blob/0.3/pufferlib/emulation.py#L192
        """
        # Default action handler does nothing. Add custom action handling here.
        return team_actions

    def features(self, obs, step):
        """Called in _poststep() via _featurize().
           See https://github.com/PufferAI/PufferLib/blob/0.3/pufferlib/emulation.py#L309
        """
        # Default featurizer pads observations to max team size
        team_features = super().features(obs, step)  # DO NOT REMOVE

        # Add custom featurization here.
        return team_features

    def rewards(self, team_rewards, team_dones, team_infos, step):
        """Called in _poststep() via _shape_rewards().
           See https://github.com/PufferAI/PufferLib/blob/0.3/pufferlib/emulation.py#L322
        """
        # The below lines update the stats and do NOT affect the reward.
        super().rewards(team_rewards, team_dones, team_infos, step)  # DO NOT REMOVE
        team_infos = {"stats": defaultdict(float)}  # DO NOT REMOVE

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.

        # Add "Healing" score based on health increase and decrease due to food and water
        health_restore = 0
        for agent_id in self.teams[self.team_id]:
            if agent_id in self.env.realm.players:
                health_restore += self.env.realm.players[agent_id].resources.health_restore
        healing_bonus = self.heal_bonus_weight if health_restore > 0 else 0

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        if self.sqrt_achievement_rewards:
            explore_bonus = math.sqrt(self._curr_unique_count) - math.sqrt(self._prev_unique_count)
        else:
            explore_bonus = min(self.clip_unique_event,
                                self._curr_unique_count - self._prev_unique_count)
        explore_bonus *= self.explore_bonus_weight

        team_reward = sum(team_rewards.values()) + explore_bonus + healing_bonus

        return team_reward, team_infos

    def infos(self, team_reward, env_done, team_done, team_infos, step):
        """Called in _poststep() via _handle_infos().
           See https://github.com/PufferAI/PufferLib/blob/0.3/pufferlib/emulation.py#L348
        """
        # The below line processes the necessary stats.
        team_infos = super().infos(team_reward, env_done, team_done, team_infos, step)  # DO NOT REMOVE

        # Add custom infos here.
        return team_infos


def create_binding(args: Namespace):
    """Create an environment binding."""

    return pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        default_args=[Config(args)],
        env_name="Neural MMO",
        suppress_env_prints=False,
        emulate_const_horizon=args.max_episode_length,
        postprocessor_cls=Postprocessor,
        postprocessor_kwargs={
            'replay_save_dir': args.replay_save_dir,
            'sqrt_achievement_rewards': args.sqrt_achievement_rewards,
            'heal_bonus_weight': args.heal_bonus_weight,
            'explore_bonus_weight': args.explore_bonus_weight
        },
    )
