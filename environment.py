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
import nmmo.systems.item as Item

from leader_board import StatPostprocessor, extract_unique_event

IMPASSIBLE = list(material.Impassible.indices)

# We can use the following mapping from task name (skill/item name as arg) to profession
TASK_TO_SKILL_MAP = {
    ":melee_": "melee",  # skils
    ":range_": "range",
    ":mage_": "mage",
    ":spear_": "melee",  # weapons
    ":bow_": "range",
    ":wand_": "mage",
    ":pickaxe_": "melee",  # tools
    ":axe_": "range",
    ":chisel_": "mage",
    ":whetstone": "melee",  # ammo
    ":arrow_": "range",
    ":runes_": "mage",
}
SKILL_LIST = sorted(list(set(skill for skill in TASK_TO_SKILL_MAP.values())))
SKILL_TO_AMMO_MAP = {
    "melee": Item.Whetstone.ITEM_TYPE_ID,
    "range": Item.Arrow.ITEM_TYPE_ID,
    "mage": Item.Runes.ITEM_TYPE_ID,
}
SKILL_TO_TILE_MAP = {
    "melee": material.Ore.index,
    "range": material.Tree.index,
    "mage": material.Crystal.index,
}
BASIC_BONUS_EVENTS = [EventCode.EAT_FOOD, EventCode.DRINK_WATER, EventCode.GO_FARTHEST]


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
                "basic_bonus_weight": args.basic_bonus_weight,
                "default_refractory_period": args.default_refractory_period,
                "death_fog_criteria": args.death_fog_criteria,
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
        basic_bonus_refractory_period=4,  # for eat, drink, progress
        basic_bonus_weight=0,
        death_fog_criteria=2,
        meander_bonus_weight=0,
        heal_bonus_weight=0,
        equipment_bonus_weight=0,
        ammofire_bonus_weight=0,
        unique_event_bonus_weight=0,
        clip_unique_event=3,
        underdog_bonus_weight = 0,
    ):
        super().__init__(env, agent_id, eval_mode, detailed_stat, early_stop_agent_num)
        self.config = env.config
        self.basic_bonus_refractory_period = basic_bonus_refractory_period
        self.basic_bonus_weight = basic_bonus_weight
        self.death_fog_criteria = death_fog_criteria
        self.meander_bonus_weight = meander_bonus_weight
        self.heal_bonus_weight = heal_bonus_weight
        self.equipment_bonus_weight = equipment_bonus_weight
        self.ammofire_bonus_weight = ammofire_bonus_weight
        self.unique_event_bonus_weight = unique_event_bonus_weight
        self.clip_unique_event = clip_unique_event
        self.underdog_bonus_weight = underdog_bonus_weight

        self._main_combat_skill = None
        self._skill_task_embedding = None

    def reset(self, obs):
        """Called at the start of each episode"""
        super().reset(obs)
        self._reset_reward_vars()
        task_name = self.env.agent_task_map[self.agent_id][0].name
        self._main_combat_skill = self._choose_combat_skill(task_name)
        self._combat_embedding = np.zeros(9, dtype=np.int16)  # copy CombatAttr to [3:]
        self._combat_embedding[SKILL_LIST.index(self._main_combat_skill)] = 1

    @staticmethod
    def _choose_combat_skill(task_name):
        task_name = task_name.lower()
        # if task_name contains specific skill or item, choose the corresponding skill
        for hint, skill in TASK_TO_SKILL_MAP.items():
            if hint in task_name:
                return skill
        # otherwise, chooose randomly
        return np.random.choice(SKILL_LIST)

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space
        # Add main combat skill (3) to the combat attr
        combat_dim = 3 + obs_space["CombatAttr"].shape[0]
        obs_space["CombatAttr"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.int16,
                                           shape=(combat_dim,))
        # Add informative tile maps: obstacle, food, water, ammo
        tile_dim = obs_space["Tile"].shape[1] + 4
        obs_space["Tile"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.int16,
                                           shape=(self.config.MAP_N_OBS, tile_dim))
        return obs_space

    def observation(self, obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        # Add main combat skill to the combat embedding
        self._combat_embedding[3:] = obs["CombatAttr"]
        obs["CombatAttr"] = self._combat_embedding

        # Add obstacle tile map to the obs
        # TODO: add entity map, update the harvest status -- how much can they help?
        obstacle = np.isin(obs["Tile"][:,2], IMPASSIBLE)
        food = obs["Tile"][:,2] == material.Foilage.index
        water = obs["Tile"][:,2] == material.Water.index
        ammo = obs["Tile"][:,2] == SKILL_TO_TILE_MAP[self._main_combat_skill]
        obs["Tile"] = np.concatenate(
            [obs["Tile"], obstacle[:,None], food[:,None], water[:,None], ammo[:,None]],
            axis=1).astype(np.int16)

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

            # Process refractory period for eat, drink, progress
            basic_bonus = 0
            self._basic_bonus_refractory_period -= 1
            for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
                if self._last_basic_events[idx] > 0:
                    if self._basic_bonus_refractory_period[idx] <= 0:
                        basic_bonus += self.basic_bonus_weight
                        self._basic_bonus_refractory_period[idx] = self.basic_bonus_refractory_period

                    # but in case under the death fog, ignore refractory period and reward running away
                    if self._curr_death_fog >= self.death_fog_criteria and \
                       event_code == EventCode.GO_FARTHEST:
                        basic_bonus += self.meander_bonus_weight  # use meander bonus

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
            reward += basic_bonus
            if self._curr_death_fog < self.death_fog_criteria:  # no extra bonus under death fog
                reward += meander_bonus + healing_bonus + equipment_bonus + ammo_fire_bonus +\
                          unique_event_bonus + underdog_bonus

        return reward, done, info

    def _reset_reward_vars(self):
        # TODO: check the effectiveness of each bonus
        # basic bonuses: eat, drink, progress
        num_basic_events = len(BASIC_BONUS_EVENTS)
        self._last_basic_events = np.zeros(num_basic_events, dtype=np.int16)
        self._basic_bonus_refractory_period = np.zeros(num_basic_events, dtype=np.int16)

        # related to death fog/progress bonus (to avoid death fog and move toward the center)
        self._curr_death_fog = 0

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
        max_offense = getattr(agent, self._main_combat_skill + "_attack")
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
        curr_tick = log[:, attr_to_col["tick"]] == self.env.realm.tick
        for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
            event_idx = curr_tick & (log[:, attr_to_col["event"]] == event_code)
            self._last_basic_events[idx] = int(sum(event_idx) > 0)
        last_ammo_idx = curr_tick & (log[:, attr_to_col["event"]] == EventCode.FIRE_AMMO) & \
                        (log[:, attr_to_col["item_type"]] == SKILL_TO_AMMO_MAP[self._main_combat_skill])
        self._last_ammo_fire = int(sum(last_ammo_idx) > 0)
        last_kill_idx = curr_tick & (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
        self._last_kill_level = max(log[last_kill_idx, attr_to_col["level"]]) if sum(last_kill_idx) > 0 else 0

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy
