from typing import Dict
import numpy as np
import gym.spaces

import nmmo
from nmmo.core.tile import TileState
from nmmo.entity.entity import EntityState
from nmmo.lib import material

EntityAttr = EntityState.State.attr_name_to_col
TileAttr = TileState.State.attr_name_to_col

NUM_FEATURE = 13  # check and match to extract_tile_feature()

DEFOGGING_VALUE = 16  # fog of war, remember for 16 ticks
VISITATION_MEMORY = 100  # visit traces, remember for 100 ticks
DEATH_FOG_CLIP = 20.  # heurisitics, the distance to death fog 20 will be clipped

TEAMMATE_REPR = 1 / 5.  # No teammate for now
PASSIVE_REPR = 2 / 5.
NEUTRAL_REPR = 3 / 5.
HOSTILE_REPR = 4 / 5.
ENEMY_REPR = 1.

DEPLETION_MAP = {
    material.Foilage.index: material.Scrub.index,
    material.Tree.index: material.Stump.index,
    material.Ore.index: material.Slag.index,
    material.Crystal.index: material.Fragment.index,
    material.Herb.index: material.Weeds.index,
    material.Fish.index: material.Ocean.index,
}

AMMO_TILE = {
    "melee": material.Ore.index,
    "range": material.Tree.index,
    "mage": material.Crystal.index,
}

class MapHelper:
    def __init__(self, config: nmmo.config.Config, agent_id,
                 profession = None,
                 img_size = 25,
                ) -> None:
        self.config = config
        self.map_size = self.config.MAP_SIZE
        self.agent_id = agent_id
        self.profession = np.random.choice(["melee", "range", "mage"]) if profession is None else profession

        # dimension of the feature map to be extracted
        self.img_size = img_size  # pixels
        self.observation_space = gym.spaces.Box(
            low=-2**15, high=2**15-1,
            shape=(NUM_FEATURE, self.img_size, self.img_size),
            dtype=np.float16
        )

        # internal memory
        self.tile_map = None
        self.fog_of_war = None
        self.fog_index = max(material.All.indices) + 1
        self.visit_map = None
        self.death_fog_map = None
        self.entity_map = None
        self.curr_pos = None

        self.x_img = np.arange(self.map_size).repeat(self.map_size)\
          .reshape(self.map_size, self.map_size)
        self.y_img = self.x_img.transpose(1, 0)

    def reset(self):
        self.tile_map = self._get_init_tile_map()
        self.fog_of_war = np.zeros((self.map_size, self.map_size))
        self.visit_map = np.zeros((self.map_size, self.map_size))
        self.death_fog_map = np.zeros((self.map_size, self.map_size))
        self.entity_map = None
        self.curr_pos = None

    def _mark_point(self, arr_2d, index_arr, value, clip=False):
        arr_2d[index_arr[:, 0], index_arr[:, 1]] = \
            np.clip(value, 0., 1.) if clip else value

    def update(self, obs: Dict):
        tile_obs = obs["Tile"]
        if np.sum(tile_obs) == 0:  # dummy obs
            return

        tile_pos = tile_obs[:, TileAttr["row"]:TileAttr["col"]+1]
        x, y = tile_pos[0]

        # update the fog of war
        self.fog_of_war = np.clip(self.fog_of_war - 1, 0, DEFOGGING_VALUE)  # decay
        self._mark_point(self.fog_of_war, tile_pos, DEFOGGING_VALUE)

        # update tile types
        tile_type = tile_obs[:, TileAttr["material_id"]]
        self.tile_map[
          x:x+self.config.PLAYER_VISION_DIAMETER,
          y:y+self.config.PLAYER_VISION_DIAMETER
        ] = tile_type.reshape(
          self.config.PLAYER_VISION_DIAMETER,
          self.config.PLAYER_VISION_DIAMETER
        )

        # update the death fog
        death_fog = tile_obs[:, 3]  # death fog is attached to the tile obs
        self.death_fog_map[
          x:x+self.config.PLAYER_VISION_DIAMETER,
          y:y+self.config.PLAYER_VISION_DIAMETER
        ] = death_fog.reshape(
          self.config.PLAYER_VISION_DIAMETER,
          self.config.PLAYER_VISION_DIAMETER
        )

        # process entity obs to update the entity and visit map
        entity_obs = obs["Entity"]
        valid_entity = entity_obs[:,EntityAttr["id"]] != 0
        entities = entity_obs[valid_entity, EntityAttr["id"]]
        ent_coords = entity_obs[valid_entity, EntityAttr["row"]:EntityAttr["col"]+1]
        self.curr_pos = ent_coords[entities == self.agent_id][0]

        # update the visit map
        self.visit_map = np.clip(self.visit_map - 1, 0, VISITATION_MEMORY)  # decay
        self._mark_point(self.visit_map,
                         ent_coords[entities == self.agent_id], VISITATION_MEMORY)

        # merging all team obs into one entity map
        entity_map = np.zeros((4, self.map_size, self.map_size))
        npc_type = entity_obs[valid_entity, EntityAttr["npc_type"]]
        self._mark_point(entity_map[0], ent_coords, npc_type == 1)  # passive npcs
        self._mark_point(entity_map[1], ent_coords, npc_type == 2)  # neutral npcs
        self._mark_point(entity_map[2], ent_coords, npc_type == 3)  # hostile npcs
        self._mark_point(entity_map[3], ent_coords,
                        np.logical_and(entities != self.agent_id, entities > 0)) # enemy
        self.entity_map = entity_map[0] * PASSIVE_REPR + entity_map[1] * NEUTRAL_REPR + \
                          entity_map[2] * HOSTILE_REPR + entity_map[3] * ENEMY_REPR

        # change tile from resource to deplete in advance
        # players will harvest resources
        for eid, pos in zip(entities, ent_coords):
            if eid > 0: # is player
                new_tile = DEPLETION_MAP.get(self.tile_map[pos[0], pos[1]])
                if new_tile is not None:
                    self.tile_map[pos[0], pos[1]] = new_tile

                # fish can be harvested from an adjacent tile, so check all adjacent tiles
                for row_offset in range(-1, 2):
                    for col_offset in range(-1, 2):
                        if self.tile_map[pos[0]+row_offset, pos[1]+col_offset] == material.Fish.index:
                            self.tile_map[pos[0]+row_offset, pos[1]+col_offset] = material.Ocean.index

    # Returns shape: (NUM_FEATURE, self.img_size, self.img_size)
    def extract_tile_feature(self):
        l, r = int(self.curr_pos[0] - self.img_size // 2), int(self.curr_pos[0] + self.img_size // 2 + 1)
        u, d = int(self.curr_pos[1] - self.img_size // 2), int(self.curr_pos[1] + self.img_size // 2 + 1)
        coord_imgs = [self.x_img[l:r, u:d] / self.map_size, self.y_img[l:r, u:d] / self.map_size]
        fog_of_war_img = self.fog_of_war[l:r, u:d] / DEFOGGING_VALUE
        death_fog_img = np.clip(self.death_fog_map[l:r, u:d], 0, np.inf) / DEATH_FOG_CLIP  # ignore safe area
        entity_img = self.entity_map[l:r, u:d]
        visit_img = self.visit_map[l:r, u:d] / VISITATION_MEMORY

        # highlight important resources
        tile_img = self.tile_map[l:r, u:d] / self.fog_index
        obstacle_img = np.isin(self.tile_map[l:r, u:d], material.Impassible.indices)
        food_img = self.tile_map[l:r, u:d] == material.Foilage.index
        water_img = self.tile_map[l:r, u:d] == material.Water.index
        ammo_img = self.tile_map[l:r, u:d] == AMMO_TILE[self.profession]
        herb_img = self.tile_map[l:r, u:d] == material.Herb.index
        fish_img = self.tile_map[l:r, u:d] == material.Fish.index

        return np.stack([*coord_imgs, fog_of_war_img, death_fog_img, entity_img, visit_img, tile_img, 
                         obstacle_img, food_img, water_img, ammo_img, herb_img, fish_img],
                         axis=0).astype(np.float16)

    def _get_init_tile_map(self):
        arr = np.zeros((self.map_size, self.map_size))  # 0: Void
        map_left = self.config.MAP_BORDER
        map_right = self.map_size - self.config.MAP_BORDER
        # mark the most outside circle of grass
        arr[map_left:map_right, map_left:map_right] = material.Grass.index
        # mark the unseen tiles
        arr[map_left+1:map_right-1, map_left+1:map_right-1] = self.fog_index
        return arr
