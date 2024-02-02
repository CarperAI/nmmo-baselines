from typing import Dict, List, Optional
from nmmo.task.task_spec import TaskSpec
import numpy as np
import dill
import json
from types import SimpleNamespace

from argparse import Namespace
import math
import copy

import nmmo
from nmmo.lib.log import EventCode
from nmmo.core.observation import Observation
from nmmo.systems.skill import Skills
from nmmo.entity.entity import Entity

import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor, calculate_entropy

_DEBUG_TASK_REWARD = False
# 指定任务内容以debug该任务的奖励设置
_DEBUG_TASK_SETTING = {}  # {"harvest": {"Fishing": 0.01}}

_EVENTS = [
    "EAT_FOOD",
    "DRINK_WATER",
    "GO_FARTHEST",
    "SCORE_HIT",
    "PLAYER_KILL",
    "CONSUME_ITEM",
    "GIVE_ITEM",
    "DESTROY_ITEM",
    "HARVEST_ITEM",
    "EQUIP_ITEM",
    "LOOT_ITEM",
    "GIVE_GOLD",
    "LIST_ITEM",
    "EARN_GOLD",
    "BUY_ITEM",
    "LEVEL_UP",
]
EVENTCODE_TO_EVENT = {getattr(EventCode, _): _ for _ in _EVENTS}
_COLS = [
    "type",
    "level",
    "number",
    "gold",
    "target_ent",
]


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
        self.RESOURCE_RESILIENT_POPULATION = args.resilient_population

        self.COMMUNICATION_SYSTEM_ENABLED = False

        self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity


class Postprocessor(StatPostprocessor):
    def __init__(
        self,
        env,
        is_multiagent,
        agent_id,
        eval_mode=False,
        early_stop_agent_num=0,
        sqrt_achievement_rewards=False,
        heal_bonus_weight=0,
        meander_bonus_weight=0,
        explore_bonus_weight=0,
        task_learning_bonus_weight=0,
        alive_bonus_weight=0,
        clip_unique_event=3,
        adjust_ori_reward=False,
        train_tasks_info=None,
        task_reward_settings=None,
        debug_print_events=False,
    ):
        super().__init__(env, agent_id, eval_mode)
        self.early_stop_agent_num = early_stop_agent_num
        self.sqrt_achievement_rewards = sqrt_achievement_rewards
        self.heal_bonus_weight = heal_bonus_weight
        self.meander_bonus_weight = meander_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

        self.adjust_ori_reward = adjust_ori_reward

        self.debug_print_events = debug_print_events

        self.alive_bonus_weight = alive_bonus_weight

        # 任务奖励相关
        self.train_tasks_info = train_tasks_info
        self._task_index: Optional[int] = None  # 当前任务在`train_tasks_info`的索引
        self.task_learning_bonus_weight = task_learning_bonus_weight
        self.task_reward_settings = task_reward_settings  # 各种任务的奖励设置
        self.task_reward_setting: Optional[Dict] = None  # 智能体当前被分配任务的奖励设置

        self.prev_done = False  # 之前的done状态

    def _reset_task_reward_state(self) -> None:
        # 见过的tile(以坐标为准)
        self._seen_tiles = {
            "co": set(),  # 坐标集合
            "last_update_tick": 0,  # 上次更新的tick
        }

        # 去过的tile(以坐标为准)
        self._been_tiles = {
            "co": set(),  # 坐标集合
            "last_update_tick": 0,  # 上次更新的tick
        }

        self._last_damage_inflicted = 0  # 上次的总造成伤害

        self._last_harvest_skill_exp = 0  # 上次的收获技能exp

        self._history_own = {}  # 记录最高拥有记录

    def reset(self, obs):
        """Called at the start of each episode"""
        super().reset(obs)

        self.prev_done = False

        if self.task_learning_bonus_weight:
            self._update_task_index(obs["Task"])
            self._reset_task_reward_state()

            setting = self._get_task_reward_setting()
            # # 打金视为必要技能
            # if True:
            #     setting = copy.deepcopy(setting)
            #     _ = setting.setdefault("log_value", {}).setdefault("EARN_GOLD", 0.0)
            #     setting["log_value"]["EARN_GOLD"] += 0.01
            self.task_reward_setting = setting

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
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
        """Called on reward, done, and info before they are returned from the environment"""
        self.env: nmmo.Env

        if self.adjust_ori_reward:
            reward = self._adjust_ori_reward(reward, done, info)
        # if reward:
        #     print(f"reward: {reward}, agent_id {self.agent_id}, done {done}, info: {info}")

        # Stop early if there are too few agents generating the training data
        if len(self.env.agents) <= self.early_stop_agent_num:
            done = True

        reward, done, info = super().reward_done_info(reward, done, info)

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.

        # Add "Healing" score based on health increase and decrease due to food and water
        healing_bonus = 0
        if self.heal_bonus_weight and self.agent_id in self.env.realm.players:
            if self.env.realm.players[self.agent_id].resources.health_restore > 0:
                healing_bonus = self.heal_bonus_weight

        # Add meandering bonus to encourage moving to various directions
        meander_bonus = 0
        if self.meander_bonus_weight and len(self._last_moves) > 5:
            move_entropy = calculate_entropy(self._last_moves[-8:])  # of last 8 moves
            meander_bonus = self.meander_bonus_weight * (move_entropy - 1)

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        explore_bonus = 0
        if self.explore_bonus_weight:
            if self.sqrt_achievement_rewards:
                explore_bonus = math.sqrt(self._curr_unique_count) - math.sqrt(
                    self._prev_unique_count
                )
            else:
                explore_bonus = min(
                    self.clip_unique_event,
                    self._curr_unique_count - self._prev_unique_count,
                )
            explore_bonus *= self.explore_bonus_weight

        # 活着就有的奖励
        alive_bonus = 0
        if self.alive_bonus_weight and not done:
            alive_bonus = self._get_alive_bonus()
            alive_bonus *= self.alive_bonus_weight

        # 不同任务对应的不同人工奖励
        task_learning_bonus = 0
        if self.task_learning_bonus_weight and not done:
            task_learning_bonus = self._get_task_learning_bonus()
            task_learning_bonus *= self.task_learning_bonus_weight

        if self.debug_print_events and done:
            self._print_agent_all_events()

        reward = reward + explore_bonus + healing_bonus + meander_bonus
        reward += alive_bonus
        reward += task_learning_bonus

        self.prev_done = done

        return reward, done, info

    def _adjust_ori_reward(self, reward, done, info) -> float:
        """调整原始reward大小
        NOTE: 原始奖励: 任务进度推进奖励是 1/进度条大小, 完成任务奖励1, 死亡奖励-1
        """
        if not reward:
            return reward

        task_infos = list(info["task"].values())
        assert len(task_infos) == 1
        task_info = task_infos[0]

        if reward == -1:
            assert done
            if task_info["completed"]:
                # 完成任务后可以死
                return -0.1
            else:
                # 加重惩罚没完成任务就死
                return -10.0

        if reward == 1:
            assert task_info["completed"]
            # 加大完成任务奖励
            return 10.0

        return reward

    @property
    def _eval_fn_name(self):
        return self.train_tasks_info.eval_fn_name[self._task_index]

    @property
    def _eval_fn_kwargs(self):
        return self.train_tasks_info.eval_fn_kwargs[self._task_index]

    def _update_task_index(self, task_embedding: np.ndarray) -> None:
        """根据 task embedding 找到 task 索引"""
        if self.eval_mode:
            self._task_index = None
            return

        assert task_embedding.shape == (4096,)

        # diff = self.train_tasks_info.embedding_mat - task_embedding
        # diff = np.sum(diff**2, axis=-1)
        # (indexes,) = np.where(diff == 0)

        (indexes,) = np.where(
            (self.train_tasks_info.embedding_mat == task_embedding).all(axis=1)
        )

        n_matched_task = len(indexes)
        assert (
            n_matched_task == 1
        ), f"{n_matched_task} task match emb ({task_embedding})"

        self._task_index = int(indexes[0])

        assert self._task_index < self.train_tasks_info.n

        # if _DEBUG_TASK_REWARD and self.agent_id <= 20:
        #     print(
        #         f"agent_id {self.agent_id}, task index {self._task_index}"
        #         f", {self.train_tasks_info.eval_fn_name[self._task_index]}"
        #         f", {self.train_tasks_info.eval_fn_kwargs[self._task_index]}"
        #     )

        return

    def _get_task_reward_setting(self) -> Dict:
        if _DEBUG_TASK_REWARD and _DEBUG_TASK_SETTING:
            return _DEBUG_TASK_SETTING

        # 当前任务内容
        _eval_fn_name = self._eval_fn_name
        _eval_fn_kwargs = self._eval_fn_kwargs

        if _eval_fn_name not in self.task_reward_settings:
            # print(f"Reward of eval fn {_eval_fn_name} not set")
            return {}

        # eval_fn大类的任务奖励设置
        eval_fn_setting: Dict = self.task_reward_settings[_eval_fn_name]
        # 具体任务内容奖励设置
        try:
            ret: Dict = eval_fn_setting[_eval_fn_kwargs[eval_fn_setting["_key"]]]
        except:
            ret: Dict = eval_fn_setting["_default"]

        return ret

    def _get_alive_bonus(self) -> float:
        ret = 0

        cur_tick = self.env.realm.tick
        entity: Entity = self.env.realm.players.entities[self.agent_id]

        # ret = cur_tick / 1024  # 最大1
        # entity.damage.val  # 被攻击所受的伤害

        # 缺乏生存资源负反馈
        # if entity.food.val == 0:
        #     ret += -0.001
        # if entity.water.val == 0:
        #     ret += -0.001

        # 低生命值负反馈
        health_lost = 100 - entity.health.val
        if health_lost > 50:
            ret += -(health_lost - 50) / 50 * 0.001

        return ret

    def _get_task_learning_bonus(self) -> float:
        ret = 0

        setting = self.task_reward_setting

        # 对各奖励提取方式计算奖励
        for reward_type, args in setting.items():
            if reward_type == "log":
                _reward = self._task_log_bonus(args)
            elif reward_type == "log_value":  # 不光看是否有，还看数值
                _reward = self._task_log_bonus(args, use_value=True)
            elif reward_type == "wander":
                _reward = self._task_wander_bonus(args)
            elif reward_type == "wander_occupy":
                _reward = self._task_wander_occupy_bonus(args)
            elif reward_type == "attack":
                _reward = self._task_attack_bonus(args)
            elif reward_type == "harvest":
                _reward = self._task_harvest_bonus(args)
            elif reward_type == "own":
                _reward = self._task_own_bonus(args)
            else:
                raise Exception(f"Invalid reward type {reward_type}")

            ret += _reward

            # if _DEBUG_TASK_REWARD and _reward and self.agent_id <= 20:
            #     print(
            #         f"agent_id {self.agent_id}, current_tick {self.env.realm.tick}"
            #         f", task learning bonus: type {reward_type}, setting {setting}, reward {_reward}"
            #         f", # players remain {len(self.env.realm.players.entities)}"
            #     )

        return ret

    def _task_log_bonus(self, args: Dict, use_value: bool = False) -> float:
        """[任务奖励]根据特定log给奖励"""
        ret = 0

        assert args

        cur_tick = self.env.realm.tick
        cur_logs = self.env.realm.event_log.get_data(
            agents=[self.agent_id], tick=cur_tick
        )

        attr_to_col = self.env.realm.event_log.attr_to_col

        for line in cur_logs:
            event_name = EVENTCODE_TO_EVENT.get(line[attr_to_col["event"]], "")

            if event_name in args:
                if use_value:
                    if event_name == "EARN_GOLD":
                        value = line[attr_to_col["gold"]]
                    else:
                        raise NotImplementedError(event_name)
                    ret += args[event_name] * value
                else:
                    ret += args[event_name]

        return ret

    def _task_wander_bonus(self, args: Dict) -> float:
        """[任务奖励]逛街奖励"""
        ret = 0

        # 每个新见tile的奖励
        per_tile = args["per_tile"]

        obs: Observation = self.env.obs[self.agent_id]
        current_tick = obs.current_tick
        visible_tiles = obs.tiles

        # 更新见过的tile
        n_new_seen_tiles = 0
        for tile in visible_tiles:
            x, y, t = tile
            if (x, y) not in self._seen_tiles["co"]:
                n_new_seen_tiles += 1
                self._seen_tiles["co"].add((x, y))
        self._seen_tiles["last_update_tick"] = current_tick

        # 首步不计
        if current_tick > 1:
            ret += n_new_seen_tiles * per_tile

        # if _DEBUG_TASK_REWARD and self.agent_id == 1:
        #     print(
        #         f"agent_id {self.agent_id}, current_tick {current_tick}"
        #         f", n_new_seen_tiles {n_new_seen_tiles}"
        #     )

        return ret

    def _task_wander_occupy_bonus(self, args: Dict) -> float:
        """[任务奖励]逛街奖励, 当去到一个新坐标给奖励"""
        ret = 0

        # 每个新见tile的奖励
        per_tile = args["per_tile"]

        entity: Entity = self.env.realm.players.entities[self.agent_id]
        current_tick = self.env.realm.tick

        # 更新见过的tile
        if entity.pos not in self._been_tiles["co"]:
            self._been_tiles["co"].add(entity.pos)
            # 首步不计
            if current_tick > 1:
                ret += per_tile

        self._been_tiles["last_update_tick"] = current_tick

        # if _DEBUG_TASK_REWARD and self.agent_id == 1:
        #     print(
        #         f"agent_id {self.agent_id}, current_tick {current_tick}"
        #         f", self._been_tiles {self._been_tiles}, +reward {ret}"
        #     )

        return ret

    def _task_attack_bonus(self, args: Dict) -> float:
        """[任务奖励]攻击奖励"""
        ret = 0

        entity = self.env.realm.players.entities[self.agent_id]  # 已死亡玩家会被剔除
        current_tick = self.env.realm.tick

        # 造成了新的伤害
        if entity.history.damage_inflicted > self._last_damage_inflicted:
            assert isinstance(entity.history.attack, dict)
            attack_style = entity.history.attack["style"]
            if attack_style in args:
                ret += args[attack_style]

            self._last_damage_inflicted = entity.history.damage_inflicted

        # if _DEBUG_TASK_REWARD and self.agent_id == 1:
        #     print(
        #         f"agent_id {self.agent_id}, current_tick {current_tick}"
        #         f", entity.history.attack {entity.history.attack}"
        #         f", entity.history.damage_inflicted {entity.history.damage_inflicted}"
        #     )

        return ret

    def _task_harvest_bonus(self, args: Dict) -> float:
        """[任务奖励]收获奖励"""
        ret = 0

        entity = self.env.realm.players.entities[self.agent_id]  # 已死亡玩家会被剔除
        skills: Skills = entity.skills
        current_tick = self.env.realm.tick

        skill_names = list(args.keys())
        assert (
            len(skill_names) == 1
        ), f"harvest reward require 1 skill but get {len(skill_names)}"
        skill_name = skill_names[0]

        if skill_name == "Fishing":
            skill = skills.fishing
        elif skill_name == "Herbalism":
            skill = skills.herbalism
        elif skill_name == "Prospecting":
            skill = skills.prospecting
        elif skill_name == "Carving":
            skill = skills.carving
        elif skill_name == "Alchemy":
            skill = skills.alchemy
        else:
            raise Exception(f"Invalid skill {skill_name}")

        cur_skill_exp = skill.exp.val
        exp_diff = cur_skill_exp - self._last_harvest_skill_exp
        if exp_diff > 0:
            ret += args[skill_name] * exp_diff
            self._last_harvest_skill_exp = cur_skill_exp

        # if _DEBUG_TASK_REWARD and self.agent_id <= 20:
        #     print(
        #         f"agent_id {self.agent_id}, current_tick {current_tick}"
        #         f", skill {skill_name}, exp {cur_skill_exp}, exp_diff {exp_diff}"
        #     )

        return ret

    def _task_own_bonus(self, args: Dict) -> float:
        """[任务奖励]收获奖励"""
        ret = 0

        entity: Entity = self.env.realm.players.entities[self.agent_id]
        current_tick = self.env.realm.tick

        packet = entity.inventory.packet()
        for item in packet["items"]:
            item_type = item["item"]
            level = item["level"]
            quantity = item["quantity"]

            reward_coef = args.get(item_type, args.get("", 0.0))
            if not reward_coef:
                continue

            if item_type not in self._history_own:
                self._history_own[item_type] = {}
            if level not in self._history_own[item_type]:
                self._history_own[item_type][level] = 0

            quantity_diff = quantity - self._history_own[item_type][level]

            if quantity_diff > 0:  # 破拥有纪录时更新记录并给奖励
                self._history_own[item_type][level] = quantity
                ret += quantity_diff * level * reward_coef

        # if _DEBUG_TASK_REWARD and ret:
        #     print(
        #         f"agent_id {self.agent_id}, current_tick {current_tick}"
        #         f", _history_own {self._history_own}, +reward {ret}"
        #     )

        return ret

    def _print_agent_all_events(self):
        print(f"== agent_id {self.agent_id}'s logs ==")
        log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        self._print_events_log(log, self.env.realm.event_log.attr_to_col)

    @staticmethod
    def _print_events_log(log, attr_to_col):
        for line in log:
            event_name = EVENTCODE_TO_EVENT.get(line[attr_to_col["event"]], "")
            tick = line[attr_to_col["tick"]]
            print(
                f"tick {tick}, event {event_name}: "
                + ", ".join([f"{_} {line[attr_to_col[_]]}" for _ in _COLS])
            )


def get_tasks_info_for_reward_setting(tasks_path: str) -> SimpleNamespace:
    with open(tasks_path, "rb") as f:
        curriculums: List[TaskSpec] = dill.load(f)

    print(f"Load {len(curriculums)} train curriculums")

    ret = SimpleNamespace(
        embedding_mat=None,  # 所有 task embedding 拼接成的矩阵
        eval_fn_name=[],
        eval_fn_kwargs=[],
        n=0,
    )

    _mat = []

    for curriculum in curriculums:
        eval_fn_kwargs = {
            key: value if isinstance(value, (str, int, float)) else value.__name__
            for key, value in curriculum.eval_fn_kwargs.items()
        }

        _mat.append(curriculum.embedding)
        ret.eval_fn_name.append(curriculum.eval_fn.__name__)
        ret.eval_fn_kwargs.append(eval_fn_kwargs)
        ret.n += 1

    ret.embedding_mat = np.vstack(_mat)

    return ret


def load_task_reward_settings(path: str) -> Dict:
    print(f"Load task reward setting {path}")
    with open(path, "r") as f:
        ret = json.load(f)
    return ret


def make_env_creator(args: Namespace):
    # TODO: Max episode length

    use_task_reward = (
        not args.eval_mode
        and args.task_reward_setting_path
        and args.task_learning_bonus_weight
    )

    # 任务信息将用于训练时设置人工奖励
    train_tasks_info = (
        get_tasks_info_for_reward_setting(args.tasks_path) if use_task_reward else None
    )
    task_reward_settings = (
        load_task_reward_settings(args.task_reward_setting_path)
        if use_task_reward
        else None
    )

    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args), seed=args.seed)
        env = pufferlib.emulation.PettingZooPufferEnv(
            env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
                "early_stop_agent_num": args.early_stop_agent_num,
                "sqrt_achievement_rewards": args.sqrt_achievement_rewards,
                "heal_bonus_weight": args.heal_bonus_weight,
                "meander_bonus_weight": args.meander_bonus_weight,
                "explore_bonus_weight": args.explore_bonus_weight,
                "task_learning_bonus_weight": args.task_learning_bonus_weight,
                "alive_bonus_weight": args.alive_bonus_weight,
                "adjust_ori_reward": args.adjust_ori_reward,
                "train_tasks_info": train_tasks_info,
                "task_reward_settings": task_reward_settings,
                "debug_print_events": args.debug_print_events,
            },
        )
        return env

    return env_creator
