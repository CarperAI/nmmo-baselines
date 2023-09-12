from typing import Optional, List
from dataclasses import dataclass
from collections import defaultdict, Counter

import math
import numpy as np

import pufferlib
import pufferlib.emulation

from nmmo.core.realm import Realm
from nmmo.lib.log import EventCode
import nmmo.systems.item as Item

@dataclass
class TeamResult:
    policy_id: str = None

    # event-log based, coming from process_event_log
    total_score: int = 0
    agent_kill_count: int = 0,
    npc_kill_count: int = 0,
    max_combat_level: int = 0,
    max_harvest_level: int = 0,
    max_damage: int = 0,
    max_progress_to_center: int = 0,
    eat_food_count: int = 0,
    drink_water_count: int = 0,
    attack_count: int = 0,
    item_harvest_count: int = 0,
    item_list_count: int = 0,
    item_buy_count: int = 0,

    # agent object based (fill these in the environment)
    # CHECK ME: perhaps create a stat wrapper for putting all stats in one place?
    time_alive: int = 0,
    earned_gold: int = 0,
    completed_task_count: int = 0,
    max_task_progress: float = 0,
    damage_received: int = 0,
    damage_inflicted: int = 0,
    ration_consumed: int = 0,
    potion_consumed: int = 0,
    melee_level: int = 0,
    range_level: int = 0,
    mage_level: int = 0,
    fishing_level: int = 0,
    herbalism_level: int = 0,
    prospecting_level: int = 0,
    carving_level: int = 0,
    alchemy_level: int = 0,

    # system-level
    n_timeout: Optional[int] = 0

    @classmethod
    def names(cls) -> List[str]:
        return [
            "total_score",
            "agent_kill_count",
            "npc_kill_count",
            "max_combat_level",
            "max_harvest_level",
            "max_damage",
            "max_progress_to_center",
            "eat_food_count",
            "drink_water_count",
            "attack_count",
            "item_equip_count",
            "item_harvest_count",
            "item_list_count",
            "item_buy_count",
            "time_alive",
            "earned_gold",
            "completed_task_count",
            "max_task_progress",
            "damage_received",
            "damage_inflicted",
            "ration_consumed",
            "potion_consumed",
            "melee_level",
            "range_level",
            "mage_level",
            "fishing_level",
            "herbalism_level",
            "prospecting_level",
            "carving_level",
            "alchemy_level",
        ]

def get_episode_result(realm: Realm, agent_id, detailed_stat=False):
    achieved, performed, event_cnt = process_event_log(realm, [agent_id], detailed_stat)
    # NOTE: Not actually a "team" result. Just a "team" of one agent
    result = TeamResult(
        policy_id = str(agent_id),  # TODO: put actual team/policy name here
        agent_kill_count = achieved["achieved/agent_kill_count"],
        npc_kill_count = achieved["achieved/npc_kill_count"],
        max_damage = achieved["achieved/max_damage"],
        max_progress_to_center = achieved["achieved/max_progress_to_center"],
        eat_food_count = event_cnt["event/eat_food"],
        drink_water_count = event_cnt["event/drink_water"],
        attack_count = event_cnt["event/score_hit"],
        item_harvest_count = event_cnt["event/harvest_item"],
        item_list_count = event_cnt["event/list_item"],
        item_buy_count = event_cnt["event/buy_item"],
    )

    return result, achieved, performed, event_cnt


class StatPostprocessor(pufferlib.emulation.Postprocessor):
    """Postprocessing actions and metrics of Neural MMO.
       Process wandb/leader board stats, and save replays.
    """
    def __init__(self, env, agent_id, eval_mode=False, detailed_stat=False):
        super().__init__(env, is_multiagent=True, agent_id=agent_id)
        self.eval_mode = eval_mode
        self.detailed_stat = detailed_stat
        self._reset_episode_stats()

    def reset(self, observation):
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.epoch_return = 0
        self.epoch_length = 0

        self._cod_attacked = 0
        self._cod_starved = 0
        self._cod_dehydrated = 0
        self._task_completed = 0
        self._max_task_progress = 0
        self._task_with_2_reward_signal = 0
        self._task_with_0p2_max_progress = 0
        self._curriculum = defaultdict(list)
        self._combat_level = []
        self._harvest_level = []
        self._prev_unique_count = 0
        self._curr_unique_count = 0

        # for agent results
        self._time_alive = 0
        self._damage_received = 0
        self._damage_inflicted = 0
        self._ration_consumed = 0
        self._potion_consumed = 0
        self._melee_level = 0
        self._range_level = 0
        self._mage_level = 0
        self._fishing_level = 0
        self._herbalism_level = 0
        self._prospecting_level = 0
        self._carving_level = 0
        self._alchemy_level = 0

        # saving actions for masking/scoring
        self._last_moves = []
        self._last_price = 0

    def _update_stats(self, agent):
        task = self.env.agent_task_map[agent.ent_id][0]
        # For each task spec, record whether its max progress and reward count
        self._curriculum[task.spec_name].append((task._max_progress, task.reward_signal_count))
        self._max_task_progress = task._max_progress
        if task.reward_signal_count >= 2:
            self._task_with_2_reward_signal = 1.0
        if task._max_progress >= 0.2:
            self._task_with_0p2_max_progress = 1.0
        if task.completed:
            self._task_completed = 1.0

        if agent.damage.val > 0:
            self._cod_attacked = 1.0
        elif agent.food.val == 0:
            self._cod_starved = 1.0
        elif agent.water.val == 0:
            self._cod_dehydrated = 1.0

        self._combat_level.append(agent.attack_level)
        self._harvest_level.append(max(
            agent.fishing_level.val,
            agent.herbalism_level.val,
            agent.prospecting_level.val,
            agent.carving_level.val,
            agent.alchemy_level.val,
        ))

        # For TeamResult
        self._time_alive += agent.history.time_alive.val
        self._damage_received += agent.history.damage_received
        self._damage_inflicted += agent.history.damage_inflicted
        self._ration_consumed += agent.ration_consumed
        self._potion_consumed += agent.poultice_consumed
        self._melee_level += agent.melee_level.val
        self._range_level += agent.range_level.val
        self._mage_level += agent.mage_level.val
        self._fishing_level += agent.fishing_level.val
        self._herbalism_level += agent.herbalism_level.val
        self._prospecting_level += agent.prospecting_level.val
        self._carving_level += agent.carving_level.val
        self._alchemy_level += agent.alchemy_level.val

    def observation(self, observation):
        # Mask out the last selected price
        observation["ActionTargets"]["Sell"]["Price"][self._last_price] = 0
        return observation

    def action(self, action):
        self._last_moves.append(action[8])  # 8 is the index for move direction
        self._last_price = action[10]  # 10 is the index for selling price
        return action

    def reward_done_info(self, reward, done, info):
        """Update stats + info and save replays."""

        # Remove the task from info. Curriculum info is processed in _update_stats()
        info.pop('task', None)

        # Count and store unique event counts for easier use
        log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        self._prev_unique_count = self._curr_unique_count
        self._curr_unique_count = len(extract_unique_event(log, self.env.realm.event_log.attr_to_col))

        # Check the last tick event
        attr_to_col = self.env.realm.event_log.attr_to_col
        last_kill_idx = (log[:, attr_to_col["tick"]] == self.env.realm.tick) & \
                        (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
        self._last_kill_level = max(log[last_kill_idx, attr_to_col["level"]]) if sum(last_kill_idx) > 0 else 0

        if not done:
            self.epoch_length += 1
            self.epoch_return += reward
            return reward, done, info

        if 'stats' not in info:
            info['stats'] = {}

        agent = self.env.realm.players.dead_this_tick.get(
            self.agent_id, self.env.realm.players.get(self.agent_id)
        )
        assert agent is not None
        self._update_stats(agent)

        info['return'] = self.epoch_return
        info['length'] = self.epoch_length

        info["stats"]["cod/attacked"] = self._cod_attacked
        info["stats"]["cod/starved"] = self._cod_starved
        info["stats"]["cod/dehydrated"] = self._cod_dehydrated
        info["stats"]["task/completed"] = self._task_completed
        info["stats"]["task/pcnt_2_reward_signal"] = self._task_with_2_reward_signal
        info["stats"]["task/pcnt_0p2_max_progress"] = self._task_with_0p2_max_progress
        info["stats"]["achieved/max_combat_level"] = max(self._combat_level)
        info["stats"]["achieved/max_harvest_level"] = max(self._harvest_level)
        info["stats"]["achieved/team_time_alive"] = self._time_alive
        info["stats"]["achieved/unique_events"] = self._curr_unique_count
        if self.detailed_stat:
            info["stats"]["skill"] = get_skill_stat(agent)
        info["curriculum"] = self._curriculum

        result, achieved, performed, _ = get_episode_result(self.env.realm, self.agent_id, self.detailed_stat)
        for key, val in list(achieved.items()) + list(performed.items()):
            info["stats"][key] = float(val)

        # Fill in the "TeamResult"
        result.max_task_progress = self._max_task_progress
        result.total_score = self._curr_unique_count
        result.time_alive = self._time_alive
        result.earned_gold = achieved["achieved/earned_gold"]
        result.completed_task_count = self._task_completed
        result.damage_received = self._damage_received
        result.damage_inflicted = self._damage_inflicted
        result.ration_consumed = self._ration_consumed
        result.potion_consumed = self._potion_consumed
        result.melee_level = self._melee_level
        result.range_level = self._range_level
        result.mage_level = self._mage_level
        result.fishing_level = self._fishing_level
        result.herbalism_level = self._herbalism_level
        result.prospecting_level = self._prospecting_level
        result.carving_level = self._carving_level
        result.alchemy_level = self._alchemy_level

        info["team_results"] = (self.agent_id, result)

        if self.eval_mode:
            # "return" is used for ranking in the eval mode, so put the task progress here
            info["return"] = self._max_task_progress  # this is 1 if done

        if self.detailed_stat and self.is_env_done():
            info["stats"].update(get_market_stat(self.env.realm))
            info["stats"].update(get_supply_stat(self.env.realm))

        return reward, done, info

    def is_env_done(self):
        # Trigger only when the episode is done, and has the lowest agent id in agents
        if self.agent_id > min(self.env.agents):
            return False
        if len(self.env.agents) <= self.early_stop_agent_num:  # early stop
            return True
        if self.env.realm.tick >= self.env.config.HORIZON:  # reached the end
            return True
        for player_id in self.env.agents:  # any alive agents?
            if player_id in self.env.realm.players:
                return False
        return True

# Event processing utilities for Neural MMO.

INFO_KEY_TO_EVENT_CODE = {
    "event/" + evt.lower(): val
    for evt, val in EventCode.__dict__.items()
    if isinstance(val, int)
}

# convert the numbers into binary (performed or not) for the key events
KEY_EVENT = [
    "eat_food",
    "drink_water",
    "score_hit",
    "player_kill",
    "consume_item",
    "harvest_item",
    "list_item",
    "buy_item",
    "fire_ammo",
]

ITEM_TYPE = {
    "all_item": [item.ITEM_TYPE_ID for item in Item.ALL_ITEM],
    "armor": [item.ITEM_TYPE_ID for item in Item.ARMOR],
    "weapon": [item.ITEM_TYPE_ID for item in Item.WEAPON],
    "tool": [item.ITEM_TYPE_ID for item in Item.TOOL],
    "ammo": [item.ITEM_TYPE_ID for item in Item.AMMUNITION],
    "consumable": [item.ITEM_TYPE_ID for item in Item.CONSUMABLE],
}

def process_event_log(realm, agent_list, detailed_stat=False, level_crit=3):
    """Process the event log and extract performed actions and achievements."""
    log = realm.event_log.get_data(agents=agent_list)
    attr_to_col = realm.event_log.attr_to_col

    # count the number of events
    event_cnt = {}
    for key, code in INFO_KEY_TO_EVENT_CODE.items():
        # count the freq of each event
        event_cnt[key] = int(sum(log[:, attr_to_col["event"]] == code))

    # record true or false for each event
    performed = {}
    for evt in KEY_EVENT:
        key = "event/" + evt
        performed[key] = event_cnt[key] > 0

    # check if tools, weapons, ammos, ammos were equipped
    for item_type, item_ids in ITEM_TYPE.items():
        if item_type in ["all_item", "consumable"]:
            continue
        key = "event/equip_" + item_type
        idx = (log[:, attr_to_col["event"]] == EventCode.EQUIP_ITEM) & \
              np.in1d(log[:, attr_to_col["item_type"]], item_ids)
        performed[key] = sum(idx) > 0

    # check if weapon was harvested
    key = "event/harvest_weapon"
    idx = (log[:, attr_to_col["event"]] == EventCode.HARVEST_ITEM) & \
          np.in1d(log[:, attr_to_col["item_type"]], ITEM_TYPE["weapon"])
    performed[key] = sum(idx) > 0

    key = "event/kill_level3_npc"
    idx = (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL) & \
          (log[:, attr_to_col["target_ent"]] < 0) & \
          (log[:, attr_to_col["level"]] >= level_crit)
    performed[key] = sum(idx) > 0

    # record important achievements
    achieved = {}

    # get progress to center
    idx = log[:, attr_to_col["event"]] == EventCode.GO_FARTHEST
    achieved["achieved/max_progress_to_center"] = \
        int(max(log[idx, attr_to_col["distance"]])) if sum(idx) > 0 else 0

    # get earned gold
    idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    achieved["achieved/earned_gold"] = int(sum(log[idx, attr_to_col["gold"]]))

    # get max damage
    idx = log[:, attr_to_col["event"]] == EventCode.SCORE_HIT
    achieved["achieved/max_damage"] = int(max(log[idx, attr_to_col["damage"]])) if sum(idx) > 0 else 0

    # get max possessed item levels: from harvesting, looting, buying
    idx = np.in1d(log[:, attr_to_col["event"]],
                  [EventCode.HARVEST_ITEM, EventCode.LOOT_ITEM, EventCode.BUY_ITEM])
    if sum(idx) > 0:
        for item_type, item_ids in ITEM_TYPE.items():
            if item_type == "all_item":
                continue
            idx_item = np.in1d(log[:, attr_to_col["item_type"]], item_ids)
            achieved["achieved/max_" + item_type + "_level"] = \
              int(max(log[idx & idx_item, attr_to_col["level"]])) if sum(idx & idx_item) > 0 else 1  # min level = 1

    # other notable achievements
    idx = (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
    achieved["achieved/agent_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] > 0)))
    achieved["achieved/npc_kill_count"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] < 0)))
    achieved["achieved/npc_level3_kill"] = int(sum(idx & (log[:, attr_to_col["target_ent"]] < 0) & 
                                                   (log[:, attr_to_col["level"]] >= level_crit)))
    achieved["achieved/ammo_fire_count"] = event_cnt["event/fire_ammo"]
    achieved["achieved/consume_item_count"] = event_cnt["event/consume_item"]

    # add item-related things
    if detailed_stat:
        own_idx = np.in1d(log[:, attr_to_col["event"]],
                          [EventCode.HARVEST_ITEM, EventCode.LOOT_ITEM, EventCode.BUY_ITEM])
        use_idx = np.in1d(log[:, attr_to_col["event"]],
                          [EventCode.CONSUME_ITEM, EventCode.EQUIP_ITEM])
        buy_idx = (log[:, attr_to_col["event"]] == EventCode.BUY_ITEM)
        for item in Item.ALL_ITEM:
            item_idx = log[:, attr_to_col["item_type"]] == item.ITEM_TYPE_ID
            level1_idx = log[:, attr_to_col["level"]] == 1
            level3_idx = log[:, attr_to_col["level"]] >= level_crit
            performed["item_" + item.__name__ + "/pcnt_owned_all_levels"] = sum(own_idx & item_idx) > 0
            performed["item_" + item.__name__ + "/pcnt_owned_level3_up"] = sum(own_idx & item_idx & level3_idx) > 0
            performed["item_" + item.__name__ + "/pcnt_used_all_levels"] = sum(use_idx & item_idx) > 0
            performed["item_" + item.__name__ + "/pcnt_used_level3_up"] = sum(use_idx & item_idx & level3_idx) > 0
            if sum(buy_idx & item_idx & level1_idx) > 0:
              performed["item_" + item.__name__ + "/purchase_price_level1"] = \
                np.mean(log[buy_idx & item_idx & level1_idx, attr_to_col["price"]])
            if sum(buy_idx & item_idx & level3_idx) > 0:
              performed["item_" + item.__name__ + "/purchase_price_level3_up"] = \
                np.mean(log[buy_idx & item_idx & level3_idx, attr_to_col["price"]])

    return achieved, performed, event_cnt

def extract_unique_event(log, attr_to_col):
    if len(log) == 0:  # no event logs
        return set()

    # mask some columns to make the event redundant
    cols_to_ignore = {
        EventCode.GO_FARTHEST: ["distance"],
        EventCode.SCORE_HIT: ["damage"],
        #EventCode.PLAYER_KILL: ["target_ent"],
        # treat each (item, level) differently but count only once 
        EventCode.CONSUME_ITEM: ["quantity"],
        EventCode.HARVEST_ITEM: ["quantity"],
        EventCode.EQUIP_ITEM: ["quantity"],
        EventCode.LOOT_ITEM: ["quantity"],
        EventCode.FIRE_AMMO: ["quantity"],
        EventCode.LIST_ITEM: ["type", "quantity", "price"],
        EventCode.BUY_ITEM: ["type", "quantity", "price"],
    }

    for code, attrs in cols_to_ignore.items():
        idx = log[:, attr_to_col["event"]] == code
        for attr in attrs:
            log[idx, attr_to_col[attr]] = 0

    # make every EARN_GOLD events unique, from looting and selling
    # idx = log[:, attr_to_col["event"]] == EventCode.EARN_GOLD
    # log[idx, attr_to_col["number"]] = log[
    #     idx, attr_to_col["tick"]
    # ].copy()  # this is a hack

    # return unique events after masking
    return set(tuple(row) for row in log[:, attr_to_col["event"]:])

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy

def get_skill_stat(agent, level_crit=3):
    skill_list = ["melee", "range", "mage", "fishing", "herbalism"]
    skill_stat = {}
    for skill in skill_list:
        skill_stat[skill + "_exp"] = getattr(agent, skill + "_exp").val
        skill_stat["pcnt_" + skill + "_level3_up"] = int(getattr(agent, skill + "_level").val >= level_crit)  # 1 or 0
    return skill_stat

def get_market_stat(realm, level_crit=3):
    # get the purchase count and total amount for all and each item type,
    # for all items, level 1 and level 3+ items
    market_stat = {}
    market_log = realm.event_log.get_data(event_code=EventCode.BUY_ITEM)
    attr_to_col = realm.event_log.attr_to_col
    item_level = {
        "all": market_log[:, attr_to_col["level"]] > 0,
        "level1_only": market_log[:, attr_to_col["level"]] == 1,
        "level3_up": market_log[:, attr_to_col["level"]] >= level_crit,
    }

    for level, level_idx in item_level.items():
        key = "market_" + level
        for item_type, item_ids in ITEM_TYPE.items():
            item_idx = np.in1d(market_log[:, attr_to_col["item_type"]], item_ids)
            market_stat[key + "/" + item_type + "_purchase_count"] = sum(item_idx & level_idx)
            market_stat[key + "/" + item_type + "_amount"] = \
                np.sum(market_log[item_idx & level_idx, attr_to_col["price"]])

    return market_stat

def get_supply_stat(realm, level_crit=3):
    supply_stat = {}
    log = realm.event_log.get_data()
    attr_to_col = realm.event_log.attr_to_col

    # level is used for both npcs and items
    level_idx = log[:, attr_to_col["level"]] >= level_crit

    # NPCs killed by players: total and level 3+
    idx = (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL) & \
          (log[:, attr_to_col["target_ent"]] < 0)
    supply_stat["supply/npc_kill_count"] = int(sum(idx))
    supply_stat["supply/npc_level3_kill_count"] = int(sum(idx & level_idx))

    # Money created from npcs & spent
    created_idx = (log[:, attr_to_col["event"]] == EventCode.LOOT_GOLD) & \
                  (log[:, attr_to_col["target_ent"]] < 0)
    supply_stat["supply/created_gold"] = int(sum(log[created_idx, attr_to_col["gold"]]))
    spent_idx = (log[:, attr_to_col["event"]] == EventCode.BUY_ITEM)
    supply_stat["supply/spent_gold"] = int(sum(log[spent_idx, attr_to_col["price"]]))

    # Items created
    for item_type, item_ids in ITEM_TYPE.items():
        if item_type == "all_item":
            continue
        item_idx = np.in1d(log[:, attr_to_col["item_type"]], item_ids)
        if item_type in ["armor", "tool"]:
            # items from npcs: armor, tool
            created_idx = (log[:, attr_to_col["event"]] == EventCode.LOOT_ITEM) & \
                          (log[:, attr_to_col["target_ent"]] < 0) & item_idx
        else:
            # items from harvest: weapon, ammo, consumable
            created_idx = (log[:, attr_to_col["event"]] == EventCode.HARVEST_ITEM) & item_idx
        supply_stat["supply/" + item_type + "_count_all"] = \
            int(np.sum(log[created_idx,attr_to_col["quantity"]]))
        supply_stat["supply/" + item_type + "_level3_count"] = \
            int(np.sum(log[created_idx & level_idx, attr_to_col["quantity"]]))

        if item_type == "ammo":
            # ammo usage from fire ammo
            fire_idx = (log[:, attr_to_col["event"]] == EventCode.FIRE_AMMO) & item_idx
            supply_stat["supply/" + item_type + "_fire_all"] = int(sum(fire_idx))
            supply_stat["supply/" + item_type + "_fire_level3"] = int(sum(fire_idx & level_idx))

    return supply_stat