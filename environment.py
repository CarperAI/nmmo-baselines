from argparse import Namespace
from collections import Counter
import math

import gym.spaces
import numpy as np

import pufferlib
import pufferlib.emulation

import nmmo
from nmmo.lib import material
from nmmo.lib.log import EventCode

from leader_board import StatPostprocessor, extract_unique_event

IMPASSIBLE = list(material.Impassible.indices)


#class Config(nmmo.config.Default):
class Config(nmmo.config.Tutorial):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.PROVIDE_DEATH_FOG_OBS = True
        self.MAP_FORCE_GENERATION = False
        self.PLAYER_N = args.num_agents
        self.HORIZON = args.max_episode_length
        self.MAP_N = args.num_maps
        self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
        self.MAP_CENTER = args.map_size
        self.NPC_N = args.num_npcs
        self.CURRICULUM_FILE_PATH = args.tasks_path
        self.TASK_EMBED_DIM = args.task_size
        self.RESOURCE_RESILIENT_POPULATION = args.resilient_population

        self.COMMUNICATION_SYSTEM_ENABLED = False

        # These affect training -- use the Tutorial config
        #self.PLAYER_DEATH_FOG = args.death_fog_tick
        #self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity


def make_env_creator(args: Namespace):
    # TODO: Max episode length
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
                "detailed_stat": args.detailed_stat,
                "early_stop_agent_num": args.early_stop_agent_num,
                "progress_bonus_weight": args.progress_bonus_weight,
                "meander_bonus_weight": args.meander_bonus_weight,
                "heal_bonus_weight": args.heal_bonus_weight,
                "equipment_bonus_weight": args.equipment_bonus_weight,
                "ammofire_bonus_weight": args.ammofire_bonus_weight,
                "unique_event_bonus_weight": args.unique_event_bonus_weight,
                #"underdog_bonus_weight": args.underdog_bonus_weight,
            },
        )
        return env
    return env_creator

class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id,
        eval_mode=False,
        detailed_stat=False,
        early_stop_agent_num=0,
        progress_bonus_weight=0,
        meander_bonus_weight=0,
        heal_bonus_weight=0,
        equipment_bonus_weight=0,
        ammofire_bonus_weight=0,
        unique_event_bonus_weight=0,
        clip_unique_event=3,
        underdog_bonus_weight = 0,
    ):
        super().__init__(env, agent_id, eval_mode, detailed_stat, early_stop_agent_num)
        self.progress_bonus_weight = progress_bonus_weight
        self.meander_bonus_weight = meander_bonus_weight
        self.heal_bonus_weight = heal_bonus_weight
        self.equipment_bonus_weight = equipment_bonus_weight
        self.ammofire_bonus_weight = ammofire_bonus_weight
        self.unique_event_bonus_weight = unique_event_bonus_weight
        self.clip_unique_event = clip_unique_event
        self.underdog_bonus_weight = underdog_bonus_weight

    def reset(self, obs):
        """Called at the start of each episode"""
        super().reset(obs)
        self._reset_reward_vars()

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space
        # Add obstacle tile map -- the org obs space is (225, 4)
        obs_space["Tile"] = gym.spaces.Box(
          low=-2**15, high=2**15-1, shape=(225, 5), dtype=np.int16)
        return obs_space

    def observation(self, obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        # Add obstacle tile map to the obs
        obstacle = np.isin(obs["Tile"][:,2], IMPASSIBLE).astype(np.int16)
        obs["Tile"] = np.concatenate([obs["Tile"], obstacle[:,None]], axis=1)

        # Mask out the last selected price
        obs["ActionTargets"]["Sell"]["Price"][self._last_price] = 0

        return obs

    def action(self, action):
        """Called before actions are passed from the model to the environment"""
        self._last_moves.append(action[8])  # 8 is the index for move direction
        self._last_price = action[10]  # 10 is the index for selling price
        return action

    def reward_done_info(self, reward, done, info):
        """Called on reward, done, and info before they are returned from the environment"""
        reward, done, info = super().reward_done_info(reward, done, info)  # DO NOT REMOVE

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.
        if not done:
            # Update the reward vars that are used to calculate the below bonuses
            agent = self.env.realm.players[self.agent_id]
            self._update_reward_vars(agent)

            # Add "Progress toward the center" bonus
            progress_bonus = 0
            self._farthest_bonus_refractory_period -= 1 if self._farthest_bonus_refractory_period > 0 else 0
            if self._last_go_farthest > 0 and self._farthest_bonus_refractory_period == 0:
                progress_bonus = self.progress_bonus_weight
                # refer to bptt horizon. the bonus is given once at max during each backprop
                self._farthest_bonus_refractory_period = 8

            # Add meandering bonus to encourage meandering yet moving toward the center
            meander_bonus = 0
            if len(self._last_moves) > 5:
              move_entropy = calculate_entropy(self._last_moves[-8:])  # of last 8 moves
              meander_bonus += self.meander_bonus_weight * (move_entropy - 1)

            # Add "Healing" score based on health increase and decrease due to food and water
            healing_bonus = self.heal_bonus_weight * float(agent.resources.health_restore > 0)

            # Add combat attribute bonus to encourage leveling up offense/defense
            equipment_bonus = self.equipment_bonus_weight * (self._new_max_offense + self._new_max_defense)

            # Add ammo fire bonus to encourage using ammo
            ammo_fire_bonus = self.ammofire_bonus_weight * self._last_ammo_fire

            # Unique event-based rewards, similar to exploration bonus
            # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
            unique_event_bonus = min(self._curr_unique_count - self._prev_unique_count,
                                     self.clip_unique_event) * self.unique_event_bonus_weight

            # Add "Underdog" bonus to encourage attacking higher level agents
            underdog_bonus = self.underdog_bonus_weight * float(self._last_kill_level > agent.attack_level)

            # sum up all the bonuses. Add most of bonus, when the agent is NOT under death fog
            reward += progress_bonus
            if self._curr_death_fog < 3:  # under a very light death fog, easy to run away yet
                reward += meander_bonus + healing_bonus + equipment_bonus + ammo_fire_bonus +\
                          unique_event_bonus + underdog_bonus

        return reward, done, info

    def _reset_reward_vars(self):
        # TODO: check the effectiveness of each bonus
        # death fog/progress bonus (to avoid death fog and move toward the center)
        self._curr_death_fog = 0
        self._last_go_farthest = 0  # if the agent broke the farthest record in the last tick
        self._farthest_bonus_refractory_period = 0

        # meander bonus (to prevent entropy collapse)
        self._last_moves = []
        self._last_price = 0  # to encourage changing price

        # healing bonus (to encourage eating food and drinking water)
        # NOTE: no separate reward var necessary, provided by the env

        # equipment, ammo-fire bonus (to level up offense/defense/ammo of the profession)
        # TODO: reward only the relevant profession
        self._max_offense = 0  # max melee/range/mage equipment offense so far
        self._new_max_offense = 0
        self._max_defense = 0  # max melee/range/mage equipment defense so far
        self._new_max_defense = 0
        self._last_ammo_fire = 0  # if an ammo was used in the last tick

        # unique event bonus (to encourage exploring new actions/items)
        self._prev_unique_count = 0
        self._curr_unique_count = 0

        # underdog bonus (to encourage attacking higher level agents)
        # NOTE: is this good? might be useful in the team setting?
        self._last_kill_level = 0

    def _update_reward_vars(self, agent):
        # From the agent
        self._curr_death_fog = self.env.realm.fog_map[agent.pos]
        max_offense = max(agent.melee_attack, agent.range_attack, agent.mage_attack)
        self._new_max_offense = 0
        if max_offense > self._max_offense:
            self._new_max_offense = 1.0 if self.env.realm.tick > 1 else 0
            self._max_offense = max_offense
        max_defense = max(agent.melee_defense, agent.range_defense, agent.mage_defense)
        self._new_max_defense = 0
        if max_defense > self._max_defense:
            self._new_max_defense = 1.0 if self.env.realm.tick > 1 else 0
            self._max_defense = max_defense

        # From the event logs
        log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        attr_to_col = self.env.realm.event_log.attr_to_col
        self._prev_unique_count = self._curr_unique_count
        self._curr_unique_count = len(extract_unique_event(log, self.env.realm.event_log.attr_to_col))
        last_farthest = (log[:, attr_to_col["tick"]] == self.env.realm.tick) & \
                        (log[:, attr_to_col["event"]] == EventCode.GO_FARTHEST)
        self._last_go_farthest = int(sum(last_farthest) > 0)
        last_ammo_idx = (log[:, attr_to_col["tick"]] == self.env.realm.tick) & \
                        (log[:, attr_to_col["event"]] == EventCode.FIRE_AMMO)
        self._last_ammo_fire = int(sum(last_ammo_idx) > 0)
        last_kill_idx = (log[:, attr_to_col["tick"]] == self.env.realm.tick) & \
                        (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
        self._last_kill_level = max(log[last_kill_idx, attr_to_col["level"]]) if sum(last_kill_idx) > 0 else 0

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy

