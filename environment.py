from argparse import Namespace
import math

import nmmo
try:
  from nmmo.core.game_api import AgentTraining
except ImportError:
  AgentTraining = None

import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor, calculate_entropy

class Config(nmmo.config.Default):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("MAP_FORCE_GENERATION", False)
        self.set("COMMUNICATION_SYSTEM_ENABLED", False)
        self.set("GAME_PACKS", [(AgentTraining, 1)] if AgentTraining else None)

        # Get values from args
        self.set("PLAYER_N", args.num_agents)
        self.set("HORIZON", args.max_episode_length)
        self.set("MAP_N", args.num_maps)
        self.set("PLAYER_DEATH_FOG", args.death_fog_tick)
        self.set("PATH_MAPS", f"{args.maps_path}/{args.map_size}/")
        self.set("MAP_CENTER", args.map_size)
        self.set("NPC_N", args.num_npcs)
        self.set("CURRICULUM_FILE_PATH", args.tasks_path)
        self.set("TASK_EMBED_DIM", args.task_size)
        self.set("RESOURCE_RESILIENT_POPULATION", args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", args.spawn_immunity)

class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id,
        eval_mode=False,
        early_stop_agent_num=0,
        sqrt_achievement_rewards=False,
        heal_bonus_weight=0,
        meander_bonus_weight=0,
        explore_bonus_weight=0,
        clip_unique_event=3,
    ):
        super().__init__(env, agent_id, eval_mode)
        self.early_stop_agent_num = early_stop_agent_num
        self.sqrt_achievement_rewards = sqrt_achievement_rewards
        self.heal_bonus_weight = heal_bonus_weight
        self.meander_bonus_weight = meander_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

    def reset(self, obs):
        '''Called at the start of each episode'''
        super().reset(obs)

    @property
    def observation_space(self):
        '''If you modify the shape of features, you need to specify the new obs space'''
        return super().observation_space

    """
    def observation(self, obs):
        '''Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        '''
        return obs

    def action(self, action):
        '''Called before actions are passed from the model to the environment'''
        return action
    """

    def reward_done_info(self, reward, done, info):
        '''Called on reward, done, and info before they are returned from the environment'''

        # Stop early if there are too few agents generating the training data
        if len(self.env.agents) <= self.early_stop_agent_num:
            done = True

        reward, done, info = super().reward_done_info(reward, done, info)

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.

        # Add "Healing" score based on health increase and decrease due to food and water
        healing_bonus = 0
        if self.agent_id in self.env.realm.players:
            if self.env.realm.players[self.agent_id].resources.health_restore > 0:
                healing_bonus = self.heal_bonus_weight

        # Add meandering bonus to encourage moving to various directions
        meander_bonus = 0
        if len(self._last_moves) > 5:
          move_entropy = calculate_entropy(self._last_moves[-8:])  # of last 8 moves
          meander_bonus = self.meander_bonus_weight * (move_entropy - 1)

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        if self.sqrt_achievement_rewards:
            explore_bonus = math.sqrt(self._curr_unique_count) - math.sqrt(self._prev_unique_count)
        else:
            explore_bonus = min(self.clip_unique_event,
                                self._curr_unique_count - self._prev_unique_count)
        explore_bonus *= self.explore_bonus_weight

        reward = reward + explore_bonus + healing_bonus + meander_bonus

        return reward, done, info


def make_env_creator(args: Namespace):
    # TODO: Max episode length
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                'eval_mode': args.eval_mode,
                'early_stop_agent_num': args.early_stop_agent_num,
                'sqrt_achievement_rewards': args.sqrt_achievement_rewards,
                'heal_bonus_weight': args.heal_bonus_weight,
                'meander_bonus_weight': args.meander_bonus_weight,
                'explore_bonus_weight': args.explore_bonus_weight,
            },
        )
        return env
    return env_creator
