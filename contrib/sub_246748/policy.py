import argparse
import torch
import torch.nn.functional as F
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models

import nmmo
from nmmo.entity.entity import EntityState

EVAL_MODE = False
# print(f"** EVAL_MODE {EVAL_MODE}")

EntityId = EntityState.State.attr_name_to_col["id"]


class ReducedModelV2(pufferlib.models.Policy):
    """Reduce observation space"""

    def __init__(self, env, input_size=256, hidden_size=256, task_size=4096):
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.tile_encoder = ReducedTileEncoder(input_size)
        self.player_encoder = ReducedPlayerEncoder(input_size, hidden_size)
        self.item_encoder = ReducedItemEncoder(input_size, hidden_size)
        self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
        self.proj_fc = torch.nn.Linear(5 * input_size, hidden_size)
        self.action_decoder = ReducedActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_observations):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            flat_observations,
            self.flat_observation_space,
            self.flat_observation_structure,
        )
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )

        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        inventory = self.inventory_encoder(item_embeddings)

        market_embeddings = self.item_encoder(env_outputs["Market"])
        market = self.market_encoder(market_embeddings)

        task = self.task_encoder(env_outputs["Task"])

        obs = torch.cat([tile, my_agent, inventory, market, task], dim=-1)
        obs = self.proj_fc(obs)

        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def no_explore_post_processing(self, logits):
        # logits shape (BS, n sub-action dim)
        max_index = torch.argmax(logits, dim=-1)
        ret = torch.full_like(logits, fill_value=-1e9)
        ret[torch.arange(logits.shape[0]), max_index] = 0

        return ret

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)

        if EVAL_MODE:
            actions = [self.no_explore_post_processing(logits) for logits in actions]
            # TODO: skip value

        return actions, value


class ReducedTileEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 32)

        self.tile_conv_1 = torch.nn.Conv2d(32, 16, 3)
        self.tile_conv_2 = torch.nn.Conv2d(16, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

    def forward(self, tile):
        # tile: row, col, material_id
        tile = tile[:, :, 2:]

        tile = self.embedding(tile.long().clip(0, 255))

        agents, tiles, features, embed = tile.shape
        tile = (
            tile.view(agents, tiles, features * embed)
            .transpose(1, 2)
            .view(agents, features * embed, 15, 15)
        )

        tile = F.relu(self.tile_conv_1(tile))
        tile = F.relu(self.tile_conv_2(tile))
        tile = tile.contiguous().view(agents, -1)
        tile = F.relu(self.tile_fc(tile))

        return tile


class ReducedPlayerEncoder(torch.nn.Module):
    """ """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        discrete_attr = [
            "id",  # pos player entity id & neg npc entity id
            "npc_type",
            "attacker_id",  # just pos player entity id
            "message",
        ]
        self.discrete_idxs = [
            EntityState.State.attr_name_to_col[key] for key in discrete_attr
        ]
        self.discrete_offset = torch.Tensor(
            [i * 256 for i in range(len(discrete_attr))]
        )

        _max_exp = 100
        _max_level = 10

        continuous_attr_and_scale = [
            ("row", 256),
            ("col", 256),
            ("damage", 100),
            ("time_alive", 1024),
            ("freeze", 3),
            ("item_level", 50),
            ("latest_combat_tick", 1024),
            ("gold", 100),
            ("health", 100),
            ("food", 100),
            ("water", 100),
            ("melee_level", _max_level),
            ("melee_exp", _max_exp),
            ("range_level", _max_level),
            ("range_exp", _max_exp),
            ("mage_level", _max_level),
            ("mage_exp", _max_exp),
            ("fishing_level", _max_level),
            ("fishing_exp", _max_exp),
            ("herbalism_level", _max_level),
            ("herbalism_exp", _max_exp),
            ("prospecting_level", _max_level),
            ("prospecting_exp", _max_exp),
            ("carving_level", _max_level),
            ("carving_exp", _max_exp),
            ("alchemy_level", _max_exp),
            ("alchemy_exp", _max_level),
        ]
        self.continuous_idxs = [
            EntityState.State.attr_name_to_col[key]
            for key, _ in continuous_attr_and_scale
        ]
        self.continuous_scale = torch.Tensor(
            [scale for _, scale in continuous_attr_and_scale]
        )

        self.embedding = torch.nn.Embedding(len(discrete_attr) * 256, 32)

        emb_dim = len(discrete_attr) * 32 + len(continuous_attr_and_scale)
        self.agent_fc = torch.nn.Linear(emb_dim, hidden_size)
        self.my_agent_fc = torch.nn.Linear(emb_dim, input_size)

    def forward(self, agents, my_id):
        # self._debug(agents)

        # Pull out rows corresponding to the agent
        agent_ids = agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        if self.discrete_offset.device != agents.device:
            self.discrete_offset = self.discrete_offset.to(agents.device)
            self.continuous_scale = self.continuous_scale.to(agents.device)

        # Embed each feature separately
        # agents shape (BS, agents, n of states)
        discrete = agents[:, :, self.discrete_idxs] + self.discrete_offset
        discrete = self.embedding(discrete.long().clip(0, 255))
        batch, item, attrs, embed = discrete.shape
        discrete = discrete.view(batch, item, attrs * embed)

        continuous = agents[:, :, self.continuous_idxs] / self.continuous_scale

        # shape (BS, agents, x)
        agent_embeddings = torch.cat([discrete, continuous], dim=-1)

        my_agent_embeddings = agent_embeddings[
            torch.arange(agents.shape[0]), row_indices
        ]

        # Project to input of recurrent size
        agent_embeddings = self.agent_fc(agent_embeddings)
        my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
        my_agent_embeddings = F.relu(my_agent_embeddings)

        return agent_embeddings, my_agent_embeddings

    def _debug(self, agents):
        agents_max, _ = torch.max(agents, dim=-2)
        agents_max, _ = torch.max(agents_max, dim=-2)
        print(f"agents_max {agents_max.tolist()}")


class ReducedItemEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.item_offset = torch.tensor([i * 256 for i in range(16)])
        self.embedding = torch.nn.Embedding(256, 32)

        self.fc = torch.nn.Linear(2 * 32 + 12, hidden_size)

        self.discrete_idxs = [1, 14]
        self.discrete_offset = torch.Tensor([2, 0])
        self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        self.continuous_scale = torch.Tensor(
            [
                10,
                10,
                10,
                100,
                100,
                100,
                40,
                40,
                40,
                100,
                100,
                100,
            ]
        )

    def forward(self, items):
        if self.discrete_offset.device != items.device:
            self.discrete_offset = self.discrete_offset.to(items.device)
            self.continuous_scale = self.continuous_scale.to(items.device)

        # Embed each feature separately
        discrete = items[:, :, self.discrete_idxs] + self.discrete_offset
        discrete = self.embedding(discrete.long().clip(0, 255))
        batch, item, attrs, embed = discrete.shape
        discrete = discrete.view(batch, item, attrs * embed)

        continuous = items[:, :, self.continuous_idxs] / self.continuous_scale

        item_embeddings = torch.cat([discrete, continuous], dim=-1)
        item_embeddings = self.fc(item_embeddings)
        return item_embeddings


class InventoryEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(12 * hidden_size, input_size)

    def forward(self, inventory):
        agents, items, hidden = inventory.shape
        inventory = inventory.view(agents, items * hidden)
        return self.fc(inventory)


class MarketEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, market):
        return self.fc(market).mean(-2)


class TaskEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, task_size):
        super().__init__()
        self.fc = torch.nn.Linear(task_size, input_size)

    def forward(self, task):
        return self.fc(task.clone())


class ReducedActionDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # order corresponding to action space
        self.sub_action_keys = [
            "attack_style",
            "attack_target",
            "market_buy",
            "inventory_destroy",
            "inventory_give_item",
            "inventory_give_player",
            "gold_quantity",
            "gold_target",
            "move",
            "inventory_sell",
            "inventory_price",
            "inventory_use",
        ]
        self.layers = torch.nn.ModuleDict(
            {
                "attack_style": torch.nn.Linear(hidden_size, 3),
                "attack_target": torch.nn.Linear(hidden_size, hidden_size),
                "market_buy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_destroy": torch.nn.Linear(hidden_size, hidden_size),
                # "inventory_give_item": torch.nn.Linear(hidden_size, hidden_size),  # TODO: useful for Inventory Management?
                # "inventory_give_player": torch.nn.Linear(hidden_size, hidden_size),
                # "gold_quantity": torch.nn.Linear(hidden_size, 99),
                # "gold_target": torch.nn.Linear(hidden_size, hidden_size),
                "move": torch.nn.Linear(hidden_size, 5),
                "inventory_sell": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_price": torch.nn.Linear(hidden_size, 99),
                "inventory_use": torch.nn.Linear(hidden_size, hidden_size),
            }
        )

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)

        return hidden

    def act_noob_action(self, key, mask):
        if key in ("inventory_give_item", "inventory_give_player", "gold_target"):
            noob_action_index = -1
        elif key in ("gold_quantity",):
            noob_action_index = 0
        else:
            raise NotImplementedError(key)

        logits = torch.full_like(mask, fill_value=-1e9)
        logits[:, noob_action_index] = 0

        return logits

    def forward(self, hidden, lookup):
        (
            player_embeddings,
            inventory_embeddings,
            market_embeddings,
            action_targets,
        ) = lookup

        embeddings = {
            "attack_target": player_embeddings,
            "market_buy": market_embeddings,
            "inventory_destroy": inventory_embeddings,
            "inventory_give_item": inventory_embeddings,
            "inventory_give_player": player_embeddings,
            "gold_target": player_embeddings,
            "inventory_sell": inventory_embeddings,
            "inventory_use": inventory_embeddings,
        }

        action_targets = {
            "attack_style": action_targets["Attack"]["Style"],
            "attack_target": action_targets["Attack"]["Target"],
            "market_buy": action_targets["Buy"]["MarketItem"],
            "inventory_destroy": action_targets["Destroy"]["InventoryItem"],
            "inventory_give_item": action_targets["Give"]["InventoryItem"],
            "inventory_give_player": action_targets["Give"]["Target"],
            "gold_quantity": action_targets["GiveGold"]["Price"],
            "gold_target": action_targets["GiveGold"]["Target"],
            "move": action_targets["Move"]["Direction"],
            "inventory_sell": action_targets["Sell"]["InventoryItem"],
            "inventory_price": action_targets["Sell"]["Price"],
            "inventory_use": action_targets["Use"]["InventoryItem"],
        }

        actions = []
        for key in self.sub_action_keys:
            mask = action_targets[key]

            if key in self.layers:
                layer = self.layers[key]
                embs = embeddings.get(key)
                if embs is not None and embs.shape[1] != mask.shape[1]:
                    b, _, f = embs.shape
                    zeros = torch.zeros([b, 1, f], dtype=embs.dtype, device=embs.device)
                    embs = torch.cat([embs, zeros], dim=1)

                action = self.apply_layer(layer, embs, mask, hidden)

            else:
                action = self.act_noob_action(key, mask)

            actions.append(action)

        return actions
