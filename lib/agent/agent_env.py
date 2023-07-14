from typing import Any, Dict, List

import gym
from pettingzoo.utils.env import AgentID, ParallelEnv

from lib.agent.agent import Agent

# Provides a wrapper around a ParallelEnv that allows for some agents
# to be controlled by existing policies, and the rest to be trained.


class AgentEnv(ParallelEnv):
  def __init__(self, env: ParallelEnv, agents: Dict[AgentID, Agent]):
    self._env = env
    self._agents = agents
    self._agent_keys = set(agents.keys())
    self._rewards = {}

    assert set(self._agents.keys()) < set(self._env.possible_agents), (
        "Agents must be a subset of the environment's possible agents"
        f"{self._agents.keys()}, {self._env.possible_agents}"
    )

    self.possible_agents = list(
        set(self._env.possible_agents) - set(self._agents.keys())
    )
    self._obs = None

  def action_space(self, agent_id: int) -> gym.Space:
    return self._env.action_space(agent_id)

  def observation_space(self, agent_id: int) -> gym.Space:
    return self._env.observation_space(agent_id)

  def reset(self, **kwargs) -> Dict[int, Any]:
    self._rewards = {id: 0 for id in self._agent_keys}
    self._obs = self._env.reset(**kwargs)
    return self._obs

  def step(self, actions: Dict[int, Dict[str, Any]]):
    actions = {
        **actions,
        **{
            agent_id: agent.act(self._obs.get(agent_id))
            for agent_id, agent in self._agents.items()
            if agent is not None
        },
    }
    self._obs, rewards, dones, infos = self._env.step(actions)

    for agent_id in self._agent_keys:
      self._rewards[agent_id] += rewards.get(agent_id, 0)

    return (
        self._filter(self._obs),
        self._filter(rewards),
        self._filter(dones),
        self._filter(infos),
    )

  def _filter(self, d: Dict[int, Any]) -> Dict[int, Any]:
    return {k: v for k, v in d.items() if k not in self._agent_keys}

  ############################################################################
  # PettingZoo API
  ############################################################################

  def render(self, mode="human"):
    return self._env.render(mode)

  @property
  def agents(self) -> List[AgentID]:
    return list(set(self._env.agents) - set(self._agent_keys))

  def close(self):
    return self._env.close()

  def seed(self, seed=None):
    return self._env.seed(seed)

  def state(self):
    return self._env.state()

  @property
  def metadata(self) -> Dict:
    return self._env.metadata
