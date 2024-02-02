from argparse import Namespace
import math
import numpy as np
import wandb

import nmmo
import pufferlib
import pufferlib.emulation

from leader_board import StatPostprocessor, calculate_entropy

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
    def __init__(self, env, is_multiagent, agent_id,
        eval_mode=False,
        early_stop_agent_num=0,
        sqrt_achievement_rewards=False,
        meander_bonus_weight=0,
        explore_bonus_weight=0,
        hp_bonus_weight=0,
        exp_bonus_weight=0,
        defense_bonus_weight=0,
        attack_bonus_weight=0,
        gold_bonus_weight=0,
        custom_bonus_scale=1,
        reset_spawn_immunity=False,
        clip_unique_event=3,
    ):
        super().__init__(env, agent_id, eval_mode)
        self.early_stop_agent_num = early_stop_agent_num
        self.sqrt_achievement_rewards = sqrt_achievement_rewards
        self.meander_bonus_weight = meander_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.hp_bonus_weight = hp_bonus_weight
        self.exp_bonus_weight = exp_bonus_weight
        self.defense_bonus_weight = defense_bonus_weight
        self.attack_bonus_weight = attack_bonus_weight
        self.gold_bonus_weight = gold_bonus_weight
        self.custom_bonus_scale = custom_bonus_scale
        self.reset_spawn_immunity = reset_spawn_immunity
        self.last_hp, self.last_exp, self.last_damage_received, self.last_damage_inflicted, self.last_gold, self.step = 100, 0, 0, 0, 0, 1
        self.clip_unique_event = clip_unique_event
        self.init_spawn_immunity = self.env.config.COMBAT_SPAWN_IMMUNITY

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

    def get_custom_reward(self):
        # Add meandering bonus to encourage moving to various directions
        meander_bonus = 0
        if len(self._last_moves) > 5:
          move_entropy = calculate_entropy(self._last_moves[-8:])  # of last 8 moves
          meander_bonus = move_entropy - 1

        # Unique event-based rewards, similar to exploration bonus
        # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
        if self.sqrt_achievement_rewards:
            explore_bonus = math.sqrt(self._curr_unique_count) - math.sqrt(self._prev_unique_count)
        else:
            explore_bonus = min(self.clip_unique_event,
                                self._curr_unique_count - self._prev_unique_count)

        if self.agent_id in self.env.realm.players:
            agent_info = self.env.realm.players[self.agent_id].packet()
            current_hp = agent_info["resource"]["health"]["val"]
            hp_bonus = current_hp - self.last_hp
            self.last_hp = current_hp

            skills = ["melee", "range", "mage", "fishing", "herbalism", "prospecting", "carving", "alchemy"]
            current_exps = np.array([agent_info["skills"][skill]["exp"] for skill in skills])
            current_exp = np.max(current_exps)
            exp_bonus = current_exp - self.last_exp
            assert current_exp >= self.last_exp, "exp bonus error"
            self.last_exp = current_exp

            current_damage_received = agent_info["history"]["damage_received"]
            equipment = agent_info["inventory"]["equipment"]
            defense = (equipment["melee_defense"] + equipment["range_defense"] + equipment["mage_defense"]) / 3
            defense_bonus = defense / 15
            self.last_damage_received = current_damage_received

            current_damage_inflicted = agent_info["history"]["damage_inflicted"]
            attack_bonus = current_damage_inflicted - self.last_damage_inflicted
            assert current_damage_inflicted >= self.last_damage_inflicted, "attack bonus error"
            self.last_damage_inflicted = current_damage_inflicted

            current_gold = self.env.realm.players[self.agent_id].gold.val
            gold_bonus = current_gold - self.last_gold
            self.last_gold = current_gold
            custom_reward = self.meander_bonus_weight * meander_bonus + self.explore_bonus_weight * explore_bonus + self.hp_bonus_weight * hp_bonus + self.exp_bonus_weight * exp_bonus + self.defense_bonus_weight * defense_bonus + self.attack_bonus_weight * attack_bonus + self.gold_bonus_weight * gold_bonus
            custom_info = {
                "bonus/explore": explore_bonus,
                "bonus/meander": meander_bonus,
                "bonus/hp": hp_bonus,
                "bonus/exp": exp_bonus,
                "bonus/defense": defense_bonus,
                "bonus/attack": attack_bonus,
                "bonus/gold": gold_bonus,
                "damage_inflicted": current_damage_inflicted,
                "damage_received": current_damage_received,
                }
        else:
            custom_reward = 0
            custom_info = dict()
        self.step += 1
        return custom_reward, custom_info

    def reward_done_info(self, reward, done, info):
        '''Called on reward, done, and info before they are returned from the environment'''

        # Stop early if there are too few agents generating the training data
        if len(self.env.agents) <= self.early_stop_agent_num:
            if self.reset_spawn_immunity:
                self.env.config.COMBAT_SPAWN_IMMUNITY = np.random.randint(0, self.init_spawn_immunity)
                self.env.realm.config.COMBAT_SPAWN_IMMUNITY = self.env.config.COMBAT_SPAWN_IMMUNITY
            done = True

        reward, done, info = super().reward_done_info(reward, done, info)
        custom_reward, custom_info = self.get_custom_reward()
        if self.agent_id in self.env.realm.players:
            info["bonus/task"] = reward
        info |= custom_info
        reward = reward + self.custom_bonus_scale * custom_reward

        if done:
            self.last_hp, self.last_exp, self.last_damage_received, self.last_damage_inflicted, self.last_gold, self.step = 100, 0, 0, 0, 0, 1

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
                'meander_bonus_weight': args.meander_bonus_weight,
                'explore_bonus_weight': args.explore_bonus_weight,
                'hp_bonus_weight': args.hp_bonus_weight,
                'exp_bonus_weight': args.exp_bonus_weight,
                'defense_bonus_weight': args.defense_bonus_weight,
                'attack_bonus_weight': args.attack_bonus_weight,
                'gold_bonus_weight': args.gold_bonus_weight,
                'custom_bonus_scale': args.custom_bonus_scale,
                'reset_spawn_immunity': args.reset_spawn_immunity,
            },
        )
        return env
    return env_creator
