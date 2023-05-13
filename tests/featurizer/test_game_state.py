import unittest
import numpy as np

import nmmo

# pylint: disable=import-error
from feature_extractor.game_state import GameState
from model.realikun.model import ModelArchitecture


class TestGameState(unittest.TestCase):
  def setUp(self):
    self.config = nmmo.config.Config()
    self.config.HORIZON = 1000
    self.team_size = 5
    self.game_state = GameState(self.config, self.team_size)

  def test_init(self):
    self.assertEqual(self.game_state.max_step, self.config.HORIZON)
    self.assertEqual(self.game_state.team_size, self.team_size)
    self.assertIsNone(self.game_state.curr_step)
    self.assertIsNone(self.game_state.curr_obs)
    self.assertIsNone(self.game_state.prev_obs)

  def test_reset(self):
    init_obs = {"player_1": "obs_1", "player_2": "obs_2"}
    self.game_state.reset(init_obs)
    self.assertEqual(self.game_state.curr_step, 0)
    self.assertEqual(self.game_state.curr_obs, init_obs)

  def test_update_advance(self):
    init_obs = {"player_1": "obs_1", "player_2": "obs_2"}
    self.game_state.reset(init_obs)

    obs = {"player_1": "obs_3", "player_2": "obs_4"}
    self.game_state.update(obs)
    self.assertEqual(self.game_state.curr_obs, obs)

    self.assertEqual(self.game_state.prev_obs, init_obs)
    self.assertEqual(self.game_state.curr_step, 1)

  def test_extract_game_feature(self):
    init_obs = {"player_1": "obs_1", "player_2": "obs_2"}
    self.game_state.reset(init_obs)

    obs = {"player_1": "obs_3", "player_2": "obs_4", "player_3": "obs_5"}
    self.game_state.update(obs)

    game_features = self.game_state.extract_game_feature(obs)
    self.assertIsInstance(game_features, np.ndarray)

    expected_n_alive = len(obs.keys())
    self.assertEqual(game_features[1], expected_n_alive / self.team_size)

    # check the feature dim
    expected_feat_n = 1 + 1 + ModelArchitecture.PROGRESS_NUM_FEATURES + self.team_size
    self.assertEqual(expected_feat_n, len(game_features))

if __name__ == '__main__':
  unittest.main()
