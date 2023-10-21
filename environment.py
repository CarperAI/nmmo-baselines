from argparse import Namespace
from collections import Counter
import math

import numpy as np
import gym.spaces

import nmmo
from nmmo.lib import material
from nmmo.lib.event_log import EventCode
import nmmo.systems.item as Item
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState
try:
  from nmmo.core.game_api import AgentTraining
except ImportError:
  AgentTraining = None

import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor, extract_unique_event

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
IMPASSIBLE = list(material.Impassible.indices)
ARMOR_LIST = [Item.Hat.ITEM_TYPE_ID, Item.Top.ITEM_TYPE_ID, Item.Bottom.ITEM_TYPE_ID]
RESTORE_ITEM = [Item.Ration.ITEM_TYPE_ID, Item.Potion.ITEM_TYPE_ID]

PASSIVE_REPR = 1  # matched to npc_type
NEUTRAL_REPR = 2
HOSTILE_REPR = 3
OTHER_AGENT_REPR = 4

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
SKILL_TO_TOOL_MAP = {
    "melee": Item.Pickaxe.ITEM_TYPE_ID,
    "range": Item.Axe.ITEM_TYPE_ID,
    "mage": Item.Chisel.ITEM_TYPE_ID,
}
SKILL_TO_WEAPON_MAP = {
    "melee": Item.Spear.ITEM_TYPE_ID,
    "range": Item.Bow.ITEM_TYPE_ID,
    "mage": Item.Wand.ITEM_TYPE_ID,
}
SKILL_TO_TILE_MAP = {
    "melee": material.Ore.index,
    "range": material.Tree.index,
    "mage": material.Crystal.index,
}
SKILL_TO_MASK = {
    "melee": np.array([1, 0, 0], dtype=np.int8),
    "range": np.array([0, 1, 0], dtype=np.int8),
    "mage": np.array([0, 0, 1], dtype=np.int8),
}
BASIC_BONUS_EVENTS = [EventCode.EAT_FOOD, EventCode.DRINK_WATER, EventCode.GO_FARTHEST]


class Config(nmmo.config.Default):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("MAP_FORCE_GENERATION", False)
        self.set("COMMUNICATION_SYSTEM_ENABLED", False)
        #self.set("NPC_ALLOW_ATTACK_OTHER_NPCS", False)  # NOTE: 2.0 default is True and does not have this attr
        self.set("GAME_PACKS", [(AgentTraining, 1)] if AgentTraining else None)

        # Get values from args
        self.set("PLAYER_N", args.num_agents)
        self.set("HORIZON", args.max_episode_length)
        self.set("MAP_N", args.num_maps)
        self.set("PLAYER_DEATH_FOG", args.death_fog_tick)
        self.set("PATH_MAPS", f"{args.maps_path}/seed_{args.seed}/")
        self.set("MAP_CENTER", args.map_size)
        self.set("NPC_N", args.num_npcs)
        self.set("CURRICULUM_FILE_PATH", args.tasks_path)
        self.set("TASK_EMBED_DIM", args.task_size)
        self.set("RESOURCE_RESILIENT_POPULATION", args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", args.spawn_immunity)

def make_env_creator(args: Namespace):
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
                "detailed_stat": args.detailed_stat,
                "early_stop_agent_num": args.early_stop_agent_num,
                # Experimental args
                "one_combat_style": args.one_combat_style,
                "augment_tile_obs": args.augment_tile_obs,
                "limit_sell_mask": args.limit_sell_mask,
                "limit_buy_mask": args.limit_buy_mask,
                "heuristic_use_mask": args.heuristic_use_mask,
                "use_new_bonus": args.use_new_bonus,
                # New bonus args
                "survival_mode_criteria": args.survival_mode_criteria,
                "get_resource_criteria": args.get_resource_criteria,
                "death_fog_criteria": args.death_fog_criteria,
                "survival_heal_weight": args.survival_heal_weight,
                "survival_resource_weight": args.survival_resource_weight,
                "get_resource_weight": args.get_resource_weight,
                "progress_bonus_weight": args.progress_bonus_weight,
                "runaway_bonus_weight": args.runaway_bonus_weight,
                "progress_refractory_period": args.progress_refractory_period,
                "meander_bonus_weight": args.meander_bonus_weight,
                "combat_bonus_weight": args.combat_bonus_weight,
                "upgrade_bonus_weight": args.upgrade_bonus_weight,
                "unique_event_bonus_weight": args.unique_event_bonus_weight,
                # Original (V1) bonus args
                "v1_heal_bonus_weight": args.v1_heal_bonus_weight,
                "v1_meander_bonus_weight": args.v1_meander_bonus_weight,
                "v1_explore_bonus_weight": args.v1_explore_bonus_weight,
            },
        )
        return env
    return env_creator


class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id,
        eval_mode=False,
        detailed_stat=False,
        early_stop_agent_num=0,
        # Experimental args
        one_combat_style=False,
        augment_tile_obs=False,
        limit_sell_mask=False,
        limit_buy_mask=False,
        heuristic_use_mask=False,
        use_new_bonus=False,
        # New bonus args
        survival_mode_criteria=35,
        get_resource_criteria=75,
        death_fog_criteria=2,
        survival_heal_weight=0,
        survival_resource_weight=0,
        get_resource_weight=0,
        progress_bonus_weight=0,
        runaway_bonus_weight=0,
        progress_refractory_period=5,
        meander_bonus_weight=0,
        combat_bonus_weight=0,
        upgrade_bonus_weight=0,
        unique_event_bonus_weight=0,
        clip_unique_event=3,
        # Original (V1) bonus args
        v1_heal_bonus_weight=0,
        v1_meander_bonus_weight=0,
        v1_explore_bonus_weight=0,
    ):
        super().__init__(env, agent_id, eval_mode, detailed_stat, early_stop_agent_num)
        self.config = env.config

        # Experimental args
        self.one_combat_style = one_combat_style
        self.augment_tile_obs = augment_tile_obs
        self.limit_sell_mask = limit_sell_mask
        self.limit_buy_mask = limit_buy_mask
        self.heuristic_use_mask = heuristic_use_mask
        self.use_new_bonus = use_new_bonus

        # Init reward vars
        self._reset_reward_vars()

        # New bonus args
        self.survival_mode_criteria = survival_mode_criteria
        self.get_resource_criteria = get_resource_criteria
        self.death_fog_criteria = death_fog_criteria
        self.survival_heal_weight = survival_heal_weight
        self.survival_resource_weight = survival_resource_weight
        self.get_resource_weight = get_resource_weight
        self.progress_bonus_weight = progress_bonus_weight
        self.runaway_bonus_weight = runaway_bonus_weight
        self.progress_refractory_period = progress_refractory_period
        self.meander_bonus_weight = meander_bonus_weight
        self.combat_bonus_weight = combat_bonus_weight
        self.upgrade_bonus_weight = upgrade_bonus_weight
        self.unique_event_bonus_weight = unique_event_bonus_weight
        self.clip_unique_event = clip_unique_event

        # Original (V1) bonus args
        self.v1_heal_bonus_weight = v1_heal_bonus_weight
        self.v1_meander_bonus_weight = v1_meander_bonus_weight
        self.v1_explore_bonus_weight = v1_explore_bonus_weight

        # New stuff
        self._main_combat_skill = None
        self._noop_inventry_item = np.zeros(self.config.ITEM_INVENTORY_CAPACITY + 1, dtype=np.int8)  # +1 for no-op
        self._noop_inventry_item[-1] = 1
        self._not_my_ammo = None
        self._ignore_items = None
        self._main_skill_items = None

        # dist map should not change from episode to episode
        self._dist_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        center = self.config.MAP_SIZE // 2
        for i in range(center):
            l, r = i, self.config.MAP_SIZE - i
            self._dist_map[l:r, l:r] = center - i - 1

        # placeholder for the entity maps
        self._entity_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)

    def reset(self, obs):
        """Called at the start of each episode"""
        super().reset(obs)
        self._reset_reward_vars()

        if self.one_combat_style:
            task_name = self.env.agent_task_map[self.agent_id][0].name
            self._main_combat_skill = self._choose_combat_skill(task_name)

            # NOTE: The items that are not used by the main combat skill are ignored
            # TODO: Revisit this
            self._not_my_ammo = [ammo for skill, ammo in SKILL_TO_AMMO_MAP.items() if skill != self._main_combat_skill]
            not_my_tool = [tool for skill, tool in SKILL_TO_TOOL_MAP.items() if skill != self._main_combat_skill]
            not_my_weapon = [weapon for skill, weapon in SKILL_TO_WEAPON_MAP.items() if skill != self._main_combat_skill]
            self._ignore_items = self._not_my_ammo + not_my_tool + not_my_weapon
            self._main_skill_items = [SKILL_TO_AMMO_MAP[self._main_combat_skill], SKILL_TO_TOOL_MAP[self._main_combat_skill],
                                    SKILL_TO_WEAPON_MAP[self._main_combat_skill]]

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
        # Add informative tile maps: entity, dist, obstacle, food, water, ammo
        if self.augment_tile_obs:
            add_dim = 6 if self.one_combat_style else 5
            tile_dim = obs_space["Tile"].shape[1] + add_dim
            obs_space["Tile"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.int16,
                                            shape=(self.config.MAP_N_OBS, tile_dim))
        return obs_space

    def observation(self, obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        if self.augment_tile_obs:
            obs["Tile"] = self._augment_tile_obs(obs)

        if self.limit_sell_mask:
            # Mask out Give, Destroy, Sell when there are less than 7 items
            # Without this, agents get rid of items and cannot learn to use and benefit from them
            num_item = sum(obs["Inventory"][:, ItemAttr["id"]] != 0)
            if num_item <= 7:
                obs["ActionTargets"]["Sell"]["InventoryItem"] = self._noop_inventry_item
                obs["ActionTargets"]["Give"]["InventoryItem"] = self._noop_inventry_item
                obs["ActionTargets"]["Destroy"]["InventoryItem"] = self._noop_inventry_item
            else:
                never_sell_items = RESTORE_ITEM
                if self._main_skill_items:
                    # TODO: consider item levels
                    never_sell_items += self._main_skill_items
                no_sale_idx = np.where(np.in1d(obs["Inventory"][:,ItemAttr["type_id"]], never_sell_items) == True)
                obs["ActionTargets"]["Sell"]["InventoryItem"][no_sale_idx] = 0

                # Do NOT give/destroy items other than (not my) ammo
                not_my_ammo = self._not_my_ammo if self.one_combat_style else list(SKILL_TO_AMMO_MAP.values())
                no_dump_idx = np.where(np.in1d(obs["Inventory"][:,ItemAttr["type_id"]], not_my_ammo) == False)
                obs["ActionTargets"]["Give"]["InventoryItem"][no_dump_idx] = 0
                obs["ActionTargets"]["Destroy"]["InventoryItem"][no_dump_idx] = 0

        if self.limit_buy_mask:
            # TODO: implement these heuristics
            # - Buy ration/potion
            # - Buy my tool or weapon that are higher level than the equipped ones but usable
            pass

        if self.heuristic_use_mask:
            obs["ActionTargets"]["Use"]["InventoryItem"] = self._heuristic_use_mask(obs)

        if self.one_combat_style:
            obs["ActionTargets"]["Attack"]["Style"] = SKILL_TO_MASK[self._main_combat_skill]

        # Turn off Attack no-op when there is a valid target
        mask = obs["ActionTargets"]["Attack"]["Target"]
        if sum(mask) > 1 and mask[-1] == 1:
            obs["ActionTargets"]["Attack"]["Target"][-1] = 0

        # Mask out the last selected price
        obs["ActionTargets"]["Sell"]["Price"][self._prev_price] = 0

        return obs

    def _augment_tile_obs(self, obs):
        dist = self._dist_map[obs["Tile"][:,0], obs["Tile"][:,1]]
        obstacle = np.isin(obs["Tile"][:,2], IMPASSIBLE)
        food = obs["Tile"][:,2] == material.Foilage.index
        water = obs["Tile"][:,2] == material.Water.index
        if self.one_combat_style:
            ammo = obs["Tile"][:,2] == SKILL_TO_TILE_MAP[self._main_combat_skill]

        # Process entity obs
        self._entity_map[:] = 0
        entity_idx = obs["Entity"][:, EntityAttr["id"]] != 0
        for entity in obs["Entity"][entity_idx]:
            if entity[EntityAttr["id"]] == self.agent_id:
                continue
            # Without this, adding the ally map hampered the agent training
            ent_pos = (entity[EntityAttr["row"]], entity[EntityAttr["col"]])
            if entity[EntityAttr["id"]] > 0:
                self._entity_map[ent_pos] = max(OTHER_AGENT_REPR, self._entity_map[ent_pos])
                # If a player is on the resource tile, assume the resource is harvested
                ent_idx = np.where((obs["Tile"][:,0] == entity[EntityAttr["row"]]) &
                                   (obs["Tile"][:,1] == entity[EntityAttr["col"]]))[0]
                food[ent_idx] = False
                if self.one_combat_style:
                    ammo[ent_idx] = False
            else:
                npc_type = entity[EntityAttr["npc_type"]]
                self._entity_map[ent_pos] = max(npc_type, self._entity_map[ent_pos])
        entity = self._entity_map[obs["Tile"][:,0], obs["Tile"][:,1]]

        maps = [obs["Tile"], entity[:,None], dist[:,None], obstacle[:,None], food[:,None], water[:,None]]
        if self.one_combat_style:
            maps.append(ammo[:,None])
        return np.concatenate(maps, axis=1).astype(np.int16)

    def _heuristic_use_mask(self, obs):
        # The mask returns 1 for all the "usable" items
        mask = obs["ActionTargets"]["Use"]["InventoryItem"]
        if sum(obs["Inventory"][:,ItemAttr["id"]] != 0) == 0:
            return mask

        # Do NOT issue "Use" on the equipped items
        equipped = np.where(obs["Inventory"][:,ItemAttr["equipped"]] == 1)
        mask[equipped] = 0

        # If any of these are equipped, do NOT issue "Use" on the same type of item unless it has higher level
        for type_id in ARMOR_LIST + self._main_skill_items if self.one_combat_style else ARMOR_LIST:
            type_idx = np.where(obs["Inventory"][:,ItemAttr["type_id"]] == type_id)
            # if there is an item of the same type that is not equipped
            if np.sum(obs["Inventory"][type_idx,ItemAttr["equipped"]]) > 0:
                mask[type_idx] = 0
                if len(type_idx[0]) > 1: # if there are more than 1 items of the same type
                    type_equipped = np.intersect1d(type_idx, equipped)
                    level_equipped = np.max(obs["Inventory"][type_equipped,ItemAttr["level"]])
                    max_level = np.max(obs["Inventory"][type_idx,ItemAttr["level"]])
                    if max_level > level_equipped:
                        # NOTE: This actions is ignored when the agent cannot equip the item due to lower level
                        use_this = np.argmax(obs["Inventory"][type_idx,ItemAttr["level"]])
                        mask[type_idx[0][use_this]] = 1

        # Ignore the items that are not used by the main combat skill
        if self._ignore_items:
            type_idx = np.where(np.in1d(obs["Inventory"][:,ItemAttr["type_id"]], self._ignore_items) == True)
            mask[type_idx] = 0

        # Use ration or potion ONLY when necessary
        starve_or_hydrate = min(self._curr_food_level, self._curr_water_level) == 0 and \
                            max(self._curr_food_level, self._curr_water_level) <= self.survival_mode_criteria
        if not starve_or_hydrate:
            type_idx = np.where(obs["Inventory"][:,ItemAttr["type_id"]] == Item.Ration.ITEM_TYPE_ID)
            mask[type_idx] = 0
        if not starve_or_hydrate and self._curr_health_level > self.survival_mode_criteria:
            type_idx = np.where(obs["Inventory"][:,ItemAttr["type_id"]] == Item.Potion.ITEM_TYPE_ID)
            mask[type_idx] = 0

        # Remove no-op when there is something to use
        if np.sum(mask) > 1:
            mask[-1] = 0

        return mask

    def action(self, action):
        """Called before actions are passed from the model to the environment"""
        self._prev_moves.append(action[8])  # 8 is the index for move direction
        self._prev_price = action[10]  # 10 is the index for selling price
        return action

    def reward_done_info(self, reward, done, info):
        """Called on reward, done, and info before they are returned from the environment"""
        reward, done, info = super().reward_done_info(reward, done, info)

        bonus = 0
        if not done:
            # Update the reward vars that are used to calculate the below bonuses
            agent = self.env.realm.players[self.agent_id]
            self._update_reward_vars(agent)
            bonus = self._new_bonus(agent) if self.use_new_bonus else self._original_bonus(agent)

        return reward + bonus, done, info

    def _original_bonus(self, agent):
        # Add "Healing" score based on health increase and decrease due to food and water
        healing_bonus = 0
        if agent.resources.health_restore > 0:
            healing_bonus = self.v1_heal_bonus_weight

        # Add meandering bonus to encourage moving to various directions
        meander_bonus = 0
        if len(self._prev_moves) > 5:
          move_entropy = calculate_entropy(self._prev_moves[-8:])  # of last 8 moves
          meander_bonus = self.v1_meander_bonus_weight * (move_entropy - 1)

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        explore_bonus = min(self.clip_unique_event, self._curr_unique_count - self._prev_unique_count)
        explore_bonus *= self.v1_explore_bonus_weight

        return healing_bonus + meander_bonus + explore_bonus

    def _original_plus_eat_bonus(self, agent):
        assert self.use_new_bonus, "use_new_bonus must be True"
        basic_bonus = 0
        for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
            if self._prev_basic_events[idx] > 0:
                if event_code == EventCode.EAT_FOOD:
                    # progress and eat
                    if self._curr_dist < self._prev_eat_dist:
                        basic_bonus += self.progress_bonus_weight
                        self._prev_eat_dist = self._curr_dist
                    # eat when starve
                    if self._prev_food_level <= self.survival_mode_criteria:
                        basic_bonus += self.survival_resource_weight
                    else:
                        basic_bonus += self.get_resource_weight

                if event_code == EventCode.DRINK_WATER:
                    # progress and drink
                    if self._curr_dist < self._prev_drink_dist:
                        basic_bonus += self.progress_bonus_weight
                        self._prev_drink_dist = self._curr_dist
                    # drink when dehydrated
                    if self._prev_water_level <= self.survival_mode_criteria:
                        basic_bonus += self.survival_resource_weight
                    else:
                        basic_bonus += self.get_resource_weight

        # Add "Healing" score based on health increase and decrease due to food and water
        healing_bonus = 0
        if agent.resources.health_restore > 0:
            healing_bonus = self.v1_heal_bonus_weight

        # Add meandering bonus to encourage moving to various directions
        meander_bonus = 0
        if len(self._prev_moves) > 5:
          move_entropy = calculate_entropy(self._prev_moves[-8:])  # of last 8 moves
          meander_bonus = self.v1_meander_bonus_weight * (move_entropy - 1)

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        explore_bonus = min(self.clip_unique_event, self._curr_unique_count - self._prev_unique_count)
        explore_bonus *= self.v1_explore_bonus_weight

        return basic_bonus + healing_bonus + meander_bonus + explore_bonus

    def _new_bonus(self, agent):
        # Survival bonus: mainly heal bonus
        # NOTE: agents got addicted to this bonus under death fog, so added death fog criteria
        survival_bonus = 0
        if self._prev_health_level <= self.survival_mode_criteria and \
            self._curr_death_fog < self.death_fog_criteria:  # not under death fog
            # 10 in case of enough food/water, 50+ for potion
            survival_bonus += self.survival_heal_weight * agent.resources.health_restore

        # Survival & progress bonuses: eat & progress, drink & progress, run away from the death fog
        progress_bonus = 0
        self._progress_refractory_counter -= 1
        for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
            if self._prev_basic_events[idx] > 0:
                if event_code == EventCode.EAT_FOOD:
                    # progress and eat
                    if self._curr_dist < self._prev_eat_dist:
                        progress_bonus += self.progress_bonus_weight
                        self._prev_eat_dist = self._curr_dist
                    # eat when starve
                    if self._prev_food_level <= self.survival_mode_criteria:
                        survival_bonus += self.survival_resource_weight
                    elif self._prev_food_level <= self.get_resource_criteria:
                        # under death fog, priotize running away
                        if self._curr_death_fog < self.death_fog_criteria:  # not under death fog
                            survival_bonus += self.get_resource_weight

                if event_code == EventCode.DRINK_WATER:
                    # progress and drink
                    if self._curr_dist < self._prev_drink_dist:
                        progress_bonus += self.progress_bonus_weight
                        self._prev_drink_dist = self._curr_dist
                    # drink when dehydrated
                    if self._prev_water_level <= self.survival_mode_criteria:
                        survival_bonus += self.survival_resource_weight
                    elif self._prev_water_level <= self.get_resource_criteria:
                        # under death fog, priotize running away
                        if self._curr_death_fog < self.death_fog_criteria:  # not under death fog
                            survival_bonus += self.get_resource_weight

                # run away from death fog
                if event_code == EventCode.GO_FARTHEST:
                    if 1 < self._curr_death_fog < self._prev_death_fog or \
                       self._progress_refractory_counter <= 0:
                        progress_bonus += self.runaway_bonus_weight
                        self._progress_refractory_counter = self.progress_refractory_period
        # Run away from death fog (can get duplicate bonus, but worth rewarding)
        if self._curr_dist < min(self._prev_dist[-8:]):
            if 1 < self._curr_death_fog < self._prev_death_fog or self._progress_refractory_counter <= 0:
                progress_bonus += self.runaway_bonus_weight
                self._progress_refractory_counter = self.progress_refractory_period

        # Add meandering bonus to encourage meandering (to prevent entropy collapse)
        meander_bonus = 0
        if len(self._prev_moves) > 5:
            move_entropy = calculate_entropy(self._prev_moves[-8:])  # of last 8 moves
            meander_bonus = self.meander_bonus_weight * (move_entropy - 1)

        # Add combat bonus to encourage combat activities that increase exp
        #combat_bonus = self.combat_bonus_weight * (self._curr_combat_exp - self._prev_combat_exp)

        # Add upgrade bonus to encourage leveling up offense/defense
        upgrade_bonus = self.upgrade_bonus_weight * (self._new_max_offense + self._new_max_defense)

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        unique_event_bonus = min(self._curr_unique_count - self._prev_unique_count,
                                 self.clip_unique_event) * self.unique_event_bonus_weight

        # Sum up all the bonuses. Under the survival mode, ignore some bonuses
        bonus = survival_bonus + progress_bonus + upgrade_bonus + meander_bonus
        if not self._survival_mode:
            bonus += unique_event_bonus

        return bonus

    def _reset_reward_vars(self):
        # Meander_bonus (to prevent entropy collapse)
        self._prev_moves = []
        self._prev_price = 0  # to encourage changing price

        # Unique event bonus (also, exploreation bonus)
        self._prev_unique_count = self._curr_unique_count = 0

        if self.use_new_bonus:
            # Survival bonus: health, food, water level
            self._prev_health_level = self._curr_health_level = 100
            self._prev_food_level = self._curr_food_level = 100
            self._prev_water_level = self._curr_water_level = 100
            self._prev_death_fog = self._curr_death_fog = 0
            self._prev_dist = []
            self._curr_dist = np.inf
            self._survival_mode = False

            # Progress bonuses: eat & progress, drink & progress, run away from the death fog
            # (reward when agents eat/drink the farthest so far)
            num_basic_events = len(BASIC_BONUS_EVENTS)
            self._prev_basic_events = np.zeros(num_basic_events, dtype=np.int16)
            self._prev_eat_dist = np.inf
            self._prev_drink_dist = np.inf
            self._progress_refractory_counter = 0

            # main combat exp (or max of the three)
            self._prev_combat_exp = self._curr_combat_exp = 0

            # equipment bonus (to level up offense/defense/ammo of the profession)
            self._max_offense = 0  # max melee/range/mage equipment offense so far
            self._new_max_offense = 0
            self._max_defense = 0  # max melee/range/mage equipment defense so far
            self._new_max_defense = 0

    def _update_reward_vars(self, agent):
        # Event log-based bonus
        log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        attr_to_col = self.env.realm.event_log.attr_to_col
        self._prev_unique_count = self._curr_unique_count
        self._curr_unique_count = len(extract_unique_event(log, attr_to_col))

        if self.use_new_bonus:
            # From the agent
            self._prev_health_level = self._curr_health_level
            self._curr_health_level = agent.resources.health.val
            self._prev_food_level = self._curr_food_level
            self._curr_food_level = agent.resources.food.val
            self._prev_water_level = self._curr_water_level
            self._curr_water_level = agent.resources.water.val
            self._prev_death_fog = self._curr_death_fog
            self._curr_death_fog = round(self.env.realm.fog_map[agent.pos]) if self._exist_fog_obs else 0  # TODO: calculate death fog
            self._prev_dist.append(self._curr_dist)
            self._curr_dist = self._dist_map[agent.pos]
            self._survival_mode = True if min(self._prev_health_level,
                                            self._prev_food_level,
                                            self._prev_water_level) <= self.survival_mode_criteria or \
                                        self._curr_death_fog >= self.death_fog_criteria \
                                        else False

            offense_dict, curr_defense = get_combat_attributes(self.config, agent)
            self._prev_combat_exp = self._curr_combat_exp
            if self.one_combat_style:
                self._curr_combat_exp = getattr(agent.skills, self._main_combat_skill).exp.val
                curr_offense = offense_dict[self._main_combat_skill]
            else:
                self._curr_combat_exp = max(agent.skills.melee.exp.val, agent.skills.range.exp.val, agent.skills.mage.exp.val)
                curr_offense = max(offense_dict.values())
            if curr_offense > self._max_offense:
                self._new_max_offense = 1.0 if self.env.realm.tick > 1 else 0
                self._max_offense = curr_offense
            self._new_max_defense = 0
            if curr_defense > self._max_defense:
                self._new_max_defense = 1.0 if self.env.realm.tick > 1 else 0
                self._max_defense = curr_defense

            # From event log
            curr_tick = log[:, attr_to_col["tick"]] == self.env.realm.tick
            for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
                event_idx = curr_tick & (log[:, attr_to_col["event"]] == event_code)
                self._prev_basic_events[idx] = int(sum(event_idx) > 0)

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy

def get_combat_attributes(config, agent):
    # defense is common for all skills
    defense = config.PROGRESSION_BASE_DEFENSE
    defense += config.PROGRESSION_LEVEL_DEFENSE * agent.level
    defense += getattr(agent.equipment, "melee_defense")

    offense = {}
    for skill in SKILL_LIST:
        base_damage = getattr(config, f"PROGRESSION_{skill.upper()}_BASE_DAMAGE")
        level_damage = getattr(config, f"PROGRESSION_{skill.upper()}_LEVEL_DAMAGE")
        level_damage *= getattr(agent.skills, skill).level.val
        equipment_damage = getattr(agent.equipment, skill + "_attack")
        offense[skill] = base_damage + level_damage + equipment_damage

    return offense, defense
