from typing import Dict, Any
import numpy as np

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.lib import material
from nmmo.systems.item import ItemState

from feature_extractor.entity_helper import EntityHelper

from model.realikun.model import ModelArchitecture

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

DEFOGGING_VALUE = 16
VISITATION_MEMORY = 100

NEARBY_DIST = 9

TEAMMATE_REPR = 1 / 5.
ENEMY_REPR = 2 / 5.
PASSIVE_REPR = 3 / 5.
NEUTRAL_REPR = 4 / 5.
HOSTILE_REPR = 1.

POISON_CLIP = 20.

DEPLETION_MAP = {
  material.Foilage.index: material.Scrub.index,
  material.Tree.index: material.Stump.index,
  material.Ore.index: material.Slag.index,
  material.Crystal.index: material.Fragment.index,
  material.Herb.index: material.Weeds.index,
  material.Fish.index: material.Ocean.index,
}

class MapHelper:
  def __init__(self, config: nmmo.config.Config,
               entity_helper: EntityHelper) -> None:
    self.config = config
    self.map_size = self.config.MAP_SIZE

    self._entity_helper = entity_helper
    self._team_size = self._entity_helper.team_size

    self.tile_map = None
    self.fog_map = None
    self.visit_map = None
    self.poison_map = None
    self.entity_map = None

    self.x_img = np.arange(self.map_size+1).repeat(self.map_size+1)\
      .reshape(self.map_size+1, self.map_size+1)
    self.y_img = self.x_img.transpose(1, 0)

  def reset(self):
    self.tile_map = self._get_init_tile_map()
    self.fog_map = np.zeros((self.map_size+1, self.map_size+1))
    self.visit_map = np.zeros((self._team_size, self.map_size+1, self.map_size+1))
    self.poison_map = self._get_init_poison_map()
    self.entity_map = None

  def update(self, obs: Dict[int, Any], curr_step: int):
    # obs for this team, key: ent_id
    self.fog_map = np.clip(self.fog_map - 1, 0, DEFOGGING_VALUE)  # decay
    self.visit_map = np.clip(self.visit_map - 1, 0, VISITATION_MEMORY)  # decay

    poison_start = self.config.PLAYER_DEATH_FOG
    if poison_start is not None:
      if (curr_step - poison_start) % self.config.PLAYER_DEATH_FOG_SPEED == 0:
        self.poison_map += 1  # poison shrinking

    entity_map = np.zeros((5, self.map_size+1, self.map_size+1))

    # pylint: disable=too-many-nested-blocks
    for ent_id, player_obs in obs.items():
      # mark tile
      tile_obs = player_obs['Tile']
      tile_pos = tile_obs[:, TileAttr["row"]:TileAttr["col"]+1].astype(int)
      tile_type = tile_obs[:, TileAttr["material_id"]].astype(int)
      self._mark_point(self.fog_map, tile_pos, DEFOGGING_VALUE)
      x, y = tile_pos[0]
      self.tile_map[
         x:x+self.config.PLAYER_VISION_DIAMETER,
         y:y+self.config.PLAYER_VISION_DIAMETER
      ] = tile_type.reshape(
         self.config.PLAYER_VISION_DIAMETER,
         self.config.PLAYER_VISION_DIAMETER
      )

      # mark team/enemy/npc
      entity_obs = player_obs['Entity']
      valid_entity = entity_obs[:,EntityAttr["id"]] != 0
      entities = entity_obs[valid_entity, EntityAttr["id"]]
      ent_in_team = [self._entity_helper.is_agent_in_team(agent_id)
                     for agent_id in entities]
      ent_coords = entity_obs[valid_entity, EntityAttr["row"]:EntityAttr["col"]+1].astype(int)
      npc_type = entity_obs[valid_entity, EntityAttr["npc_type"]].astype(int)

      # merging all team obs into one entity map
      self._mark_point(entity_map[0], ent_coords, ent_in_team) # teammates
      self._mark_point(entity_map[1], ent_coords,
                       np.logical_and(np.logical_not(ent_in_team), entities > 0)) # enemy
      self._mark_point(entity_map[2], ent_coords, npc_type == -1)  # passive npcs
      self._mark_point(entity_map[3], ent_coords, npc_type == -2)  # neutral npcs
      self._mark_point(entity_map[4], ent_coords, npc_type == -3)  # hostile npcs

      # update visit map
      self._mark_point(self.visit_map[self._entity_helper.agent_id_to_pos(ent_id)],
                       ent_coords[entities == ent_id],
                       VISITATION_MEMORY)

      # change tile from resource to deplete in advance
      # players will harvest resources
      for eid, pos in zip(entities, ent_coords):
        if eid > 0: # is player
          new_tile = DEPLETION_MAP.get(self.tile_map[pos[0], pos[1]])
          if new_tile is not None:
            self.tile_map[pos[0], pos[1]] = new_tile

            # fish can be harvested from an adjacent tile
            # TODO: this part needs to be tested
            # pylint: disable=too-many-nested-blocks
            for row_offset in range(-1, 2):
              for col_offset in range(-1, 2):
                if self.tile_map[pos[0]+row_offset, pos[1]+col_offset] == material.Fish.index:
                  self.tile_map[pos[0]+row_offset, pos[1]+col_offset] = material.Ocean.index

    # update the entity map based on all team obs
    self.entity_map = entity_map[0] * TEAMMATE_REPR + entity_map[1] * ENEMY_REPR + \
      entity_map[2] * PASSIVE_REPR + entity_map[3] * NEUTRAL_REPR + entity_map[4] * HOSTILE_REPR

  # Returns shape: (TEAM_SIZE, NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
  def extract_tile_feature(self):
    img_size = ModelArchitecture.TILE_IMG_SIZE[0]
    dummy_img_feat = np.zeros((ModelArchitecture.TILE_NUM_CHANNELS, img_size, img_size))
    imgs = []
    for member_pos in range(self._team_size):
      if not self._entity_helper.is_pos_alive(member_pos):
        imgs.append(dummy_img_feat)
        continue

      curr_pos = self._entity_helper.member_location[member_pos]
      l, r = int(curr_pos[0] - img_size // 2), int(curr_pos[0] + img_size // 2 + 1)
      u, d = int(curr_pos[1] - img_size // 2), int(curr_pos[1] + img_size // 2 + 1)
      tile_img = self.tile_map[l:r, u:d] / (1 + max(material.All.indices))
      entity_img = self.entity_map[l:r, u:d]

      # CHECK ME: '/ .20' is in several places. Why do we need this?
      # poison_map increases by one at every 16 ticks
      poison_img = np.clip(self.poison_map[l:r, u:d], 0, np.inf) / POISON_CLIP

      fog_img = self.fog_map[l:r, u:d] / DEFOGGING_VALUE
      visit_img = self.visit_map[member_pos][l:r, u:d] / VISITATION_MEMORY
      coord_imgs = [self.x_img[l:r, u:d] / self.map_size, self.y_img[l:r, u:d] / self.map_size]

      # NOTE: realikun also considered obstacle_img, view_img
      img = np.stack([tile_img, entity_img, poison_img, fog_img, visit_img, *coord_imgs])
      imgs.append(img)

    return np.stack(imgs).astype(np.float32)

  def nearby_features(self, row: int, col: int, nearby_dist=NEARBY_DIST):
    # CHECK ME(kywch): my understanding of this function is to provide
    #   especially important spatial features to the agent
    #   such as nearby food, water, herb (poultice), fish (ration), obstacles, poison
    #   As long as the order is fixed, agents will make sense of and use these features
    near_tile_map = self.tile_map[row-nearby_dist//2:row+(nearby_dist//2+1),
                                  col-nearby_dist//2:col+(nearby_dist//2+1)]
    feat_arr = []
    for i in range(nearby_dist):
      for j in range(nearby_dist):
        # (i,j) = (4,4) is the provided (row, col)
        if abs(i-nearby_dist//2) + abs(j-nearby_dist//2) <= nearby_dist//2:
          feat_arr.append(near_tile_map[i, j] == material.Foilage.index) # food_arr
          feat_arr.append(near_tile_map[i, j] == material.Water.index) # water_arr
          feat_arr.append(near_tile_map[i, j] == material.Herb.index) # herb_arr
          feat_arr.append(near_tile_map[i, j] == material.Fish.index) # fish_arr
          feat_arr.append(near_tile_map[i, j] in material.Impassible.indices) # obstacle_arr
          feat_arr.append(max(0, self.poison_map[row+i, col+j]) / POISON_CLIP) # poison map

    return np.array(feat_arr, dtype=np.float32)

  def dummy_nearby_features(self):
    return np.zeros(ModelArchitecture.NEARBY_NUM_FEATURES, dtype=np.float32)

  def legal_moves(self, obs):
    assert self.config.PROVIDE_ACTION_TARGETS,\
      "config.PROVIDE_ACTION_TARGETS must be set True"
    moves = np.zeros((self._team_size, len(nmmo.action.Direction.edges)), dtype=np.float32)
    for member_pos in range(self._team_size):
      ent_id = self._entity_helper.pos_to_agent_id(member_pos)
      if ent_id in obs:
        moves[member_pos] = obs[ent_id]["ActionTargets"][nmmo.action.Move][nmmo.action.Direction]

    return moves

  def _get_init_tile_map(self):
    arr = np.zeros((self.map_size+1, self.map_size+1))
    # mark the most outside circle of grass
    map_left = self.config.MAP_BORDER
    map_right = self.map_size - self.config.MAP_BORDER
    arr[map_left:map_right+1, map_left:map_right+1] = 2
    # mark the unseen tiles
    arr[map_left+1:map_right, map_left+1:map_right] = max(material.All.indices) + 1
    return arr

  def _get_init_poison_map(self):
    arr = np.ones((self.map_size + 1, self.map_size + 1))
    for i in range(self.map_size // 2):
      l, r = i + 1, self.map_size - i
      arr[l:r, l:r] = -i
    # positive value represents the poison strength
    # negative value represents the shortest distance to poison area

    # mark the safe area
    center = self.map_size // 2
    safe = self.config.PLAYER_DEATH_FOG_FINAL_SIZE
    arr[center-safe:center+safe+1, center-safe:center+safe+1] = -np.inf

    return arr

  def _mark_point(self, arr_2d, index_arr, value, clip=False):
    arr_2d[index_arr[:, 0], index_arr[:, 1]] = \
      np.clip(value, 0., 1.) if clip else value
