from argparse import ArgumentParser, Namespace
from collections import defaultdict

import nmmo
import pufferlib
import pufferlib.emulation
from nmmo.render.replay_helper import FileReplayHelper
from typing import Any, Dict


def add_args(parser: ArgumentParser):
  parser.add_argument(
      "--env.num_agents",
      dest="num_agents",
      type=int,
      default=128,
      help="number of agents to use for training (default: 128)",
  )
  parser.add_argument(
      "--env.num_npcs",
      dest="num_npcs",
      type=int,
      default=0,
      help="number of NPCs to use for training (default: 256)",
  )
  parser.add_argument(
      "--env.max_episode_length",
      dest="max_episode_length",
      type=int,
      default=1024,
      help="number of steps per episode (default: 1024)",
  )
  parser.add_argument(
      "--env.death_fog_tick",
      dest="death_fog_tick",
      type=int,
      default=None,
      help="number of ticks before death fog starts (default: None)",
  )
  parser.add_argument(
      "--env.num_maps",
      dest="num_maps",
      type=int,
      default=128,
      help="number of maps to use for training (default: 1)",
  )
  parser.add_argument(
      "--env.maps_path",
      dest="maps_path",
      type=str,
      default="maps/train/",
      help="path to maps to use for training (default: None)",
  )
  parser.add_argument(
      "--env.map_size",
      dest="map_size",
      type=int,
      default=128,
      help="size of maps to use for training (default: 128)",
  )
  parser.add_argument(
      "--env.tasks_path",
      dest="tasks_path",
      type=str,
      default=None,
      help="path to tasks to use for training (default: tasks.pkl)",
  )


class Config(
    nmmo.config.Medium,
    nmmo.config.Terrain,
    nmmo.config.Resource,
    nmmo.config.Progression,
    nmmo.config.Equipment,
    nmmo.config.Item,
    nmmo.config.Exchange,
    nmmo.config.Combat,
    nmmo.config.NPC,
):
  def __init__(self, args: Namespace):
    super().__init__()

    self.PROVIDE_ACTION_TARGETS = True
    self.MAP_FORCE_GENERATION = False
    self.PLAYER_N = args.num_agents
    self.HORIZON = args.max_episode_length
    self.MAP_N = args.num_maps
    self.PLAYER_DEATH_FOG = args.death_fog_tick
    self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
    self.MAP_CENTER = args.map_size
    self.NPC_N = args.num_npcs
    self.CURRICULUM_FILE_PATH = args.tasks_path


class Postprocessor(pufferlib.emulation.Postprocessor):
  def __init__(self, env, teams, team_id, replay_save_dir=None):
    super().__init__(env, teams, team_id)
    self._replay_save_dir = replay_save_dir
    if self._replay_save_dir is not None:
      self._replay_helper = FileReplayHelper()
      env.realm.record_replay(self._replay_helper)

  # def reset(self, team_obs):
  #   if self.realm.tick and self._replay_helper is not None:
  #     ReplayEnv.num_replays_saved += 1
  #     self._replay_helper.save(
  #         f"{self._replay_save_dir}/{ReplayEnv.num_replays_saved}",
  #         compress=False,
  #     )
  #   super().reset(team_obs)

  def rewards(self, team_rewards, team_dones, team_infos, step):
    agents = list(set(team_rewards.keys()).union(set(team_dones.keys())))

    team_reward = sum(team_rewards.values())
    team_info = {"stats": defaultdict(float)}

    for agent_id in agents:
      agent = self.env.realm.players.dead_this_tick.get(
          agent_id, self.env.realm.players.get(agent_id)
      )

      if agent is None:
        continue

      if agent_id in team_dones and team_dones[agent_id] is True:
        if agent.damage.val > 0:
          team_info["stats"]["cod/attacked"] += 1
        elif agent.food.val == 0:
          team_info["stats"]["cod/starved"] += 1
        elif agent.water.val == 0:
          team_info["stats"]["cod/dehydrated"] += 1
    return team_reward, team_info

  def infos(self, team_reward, env_done, team_done, team_infos, step):
    team_infos = super().infos(team_reward, env_done, team_done, team_infos, step)

    return team_infos

  # def features(self, obs, step):
  #   # for ob in obs.values():
  #   #   ob["featurized"] = self._feature_extractor(obs)
  #   return obs

  # def actions(self, actions, step):
  #   return self._feature_extractor.translate_actions(actions)


def create_binding(args: Namespace):
  return pufferlib.emulation.Binding(
      env_cls=nmmo.Env,
      default_args=[Config(args)],
      env_name="Neural MMO",
      suppress_env_prints=False,
      emulate_const_horizon=args.max_episode_length,
      postprocessor_cls=Postprocessor,
      postprocessor_args=[],
  )
