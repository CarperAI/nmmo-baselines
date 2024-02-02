from torch.nn import MultiheadAttention
import argparse
import torch
import torch.nn.functional as F
from typing import Dict

import pufferlib
import pufferlib.emulation
import pufferlib.models
#from reinforcement_learning.model_util import PopArt
import nmmo
from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class BaselineNew(pufferlib.models.Policy):
    def __init__(self, env, input_size=256, hidden_size=256, task_size=4096):
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.item_encoder = ItemEncoder(input_size, hidden_size)
        self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
        self.proj_fc = torch.nn.Linear(5 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.value_head = PopArt(hidden_size, 1, device=self.device)

    def encode_observations(self, flat_observations):
        env_outputs = pufferlib.emulation.unpack_batched_obs(flat_observations,
                                                             self.flat_observation_space, self.flat_observation_structure)
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

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value

    def forward(self, env_outputs):
        '''Forward pass for PufferLib compatibility'''
        hidden, lookup = self.encode_observations(env_outputs)
        actions, value = self.decode_actions(hidden, lookup)

        return actions, value


class SelfAttention(torch.nn.Module):
    """
    Implementation of Self Attention Layer from 
    https://arxiv.org/pdf/2305.17352.pdf
    """

    def __init__(self, input_size, heads=4, embed_size=32):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = torch.nn.Linear(
            self.input_size, self.emb_size * heads, bias=False)
        self.toqueries = torch.nn.Linear(
            self.input_size, self.emb_size * heads, bias=False)
        self.tovalues = torch.nn.Linear(
            self.input_size, self.emb_size * heads, bias=False)

    def forward(self, x):
        b, t, hin = x.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'

        h = self.heads
        e = self.emb_size

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        self.dot = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        values = values.view(b, h, t, e)
        values = values.transpose(1, 2).contiguous().view(b, t, h * e)
        self.values = values
        return out


class TileEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.tile_offset = torch.tensor([i * 256 for i in range(3)])
        self.embedding = torch.nn.Embedding(3 * 256, 32)

        self.multihead_attn = MultiheadAttention(
            16, 4)  # Added MultiheadAttention layer

        self.tile_conv_1 = torch.nn.Conv2d(96, 64, 3)
        self.tile_conv_2 = torch.nn.Conv2d(64, 32, 3)
        self.tile_conv_3 = torch.nn.Conv2d(32, 16, 3)
        self.tile_fc = torch.nn.Linear(16 * 9 * 9, input_size)
        self.activation = torch.nn.SiLU()

    def forward(self, tile):
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7
        tile = self.embedding(
            tile.long().clip(0, 255) + self.tile_offset.to(tile.device)
        )

        agents, tiles, features, embed = tile.shape
        tile = (
            tile.view(agents, tiles, features * embed)
            .transpose(1, 2)
            .view(agents, features * embed, 15, 15)
        )

        tile = self.activation(self.tile_conv_1(tile))
        tile = self.activation(self.tile_conv_2(tile))
        tile = self.activation(self.tile_conv_3(tile))  # Additional layer
        # Reshape for MultiheadAttention
        tile = tile.view(tile.size(0), tile.size(1), -1).permute(2, 0, 1)
        tile, _ = self.multihead_attn(
            tile, tile, tile)  # Apply MultiheadAttention
        tile = tile.permute(1, 2, 0).view(agents, 16, 9, 9)  # Reshape back
        tile = tile.contiguous().view(agents, -1)
        tile = self.activation(self.tile_fc(tile))

        return tile


class PlayerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.entity_dim = 31
        self.player_offset = torch.tensor(
            [i * 256 for i in range(self.entity_dim)])
        self.embedding = torch.nn.Embedding(self.entity_dim * 256, 32)

        self.agent_fc = torch.nn.Linear(self.entity_dim * 32, hidden_size)
        self.my_agent_fc = torch.nn.Linear(self.entity_dim * 32, input_size)

    def forward(self, agents, my_id):
        # Pull out rows corresponding to the agent
        agent_ids = agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(
                dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        agent_embeddings = self.embedding(
            agents.long().clip(0, 255) + self.player_offset.to(agents.device)
        )
        batch, agent, attrs, embed = agent_embeddings.shape

        # Embed each feature separately
        agent_embeddings = agent_embeddings.view(batch, agent, attrs * embed)
        my_agent_embeddings = agent_embeddings[
            torch.arange(agents.shape[0]), row_indices
        ]

        # Project to input of recurrent size
        agent_embeddings = self.agent_fc(agent_embeddings)
        my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
        my_agent_embeddings = F.silu(my_agent_embeddings)

        return agent_embeddings, my_agent_embeddings


class ItemEncoder(torch.nn.Module):
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
                1 / 10,
                1 / 10,
                1 / 10,
                1 / 100,
                1 / 100,
                1 / 100,
                1 / 40,
                1 / 40,
                1 / 40,
                1 / 100,
                1 / 100,
                1 / 100,
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


class ActionDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "attack_style": torch.nn.Linear(hidden_size, 3),
                "attack_target": torch.nn.Linear(hidden_size, hidden_size),
                "market_buy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_destroy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_give_item": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_give_player": torch.nn.Linear(hidden_size, hidden_size),
                "gold_quantity": torch.nn.Linear(hidden_size, 99),
                "gold_target": torch.nn.Linear(hidden_size, hidden_size),
                "move": torch.nn.Linear(hidden_size, 5),
                "inventory_sell": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_price": torch.nn.Linear(hidden_size, 99),
                "inventory_use": torch.nn.Linear(hidden_size, hidden_size),
            }
        )
        self.attn = SelfAttention(hidden_size, 4, hidden_size//4)
        self.fc = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.rnn = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=5,
            batch_first=True,
        )
        self.prev_player_states = None
        self.prev_inventory_states = None
        self.prev_hidden_states = None

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)

        return hidden

    def forward(self, hidden, lookup):
        (
            player_embeddings,
            inventory_embeddings,
            market_embeddings,
            action_targets,
        ) = lookup
        # player_embeddings.shape:  torch.Size([768, 100, 256])
        # inventory_embeddings.shape:  torch.Size([768, 12, 256])
        # market_embeddings.shape:  torch.Size([768, 1024, 256])
        # hidden.shape:  torch.Size([768, 256])

        player_embeddings_before = player_embeddings.clone()
        inventory_embeddings_before = inventory_embeddings.clone()
        hidden_before = hidden.clone()
        hidden = hidden.unsqueeze(1)
        # Check for the prev states shape
        if self.prev_player_states is None or self.prev_player_states[0].shape != (self.rnn.num_layers, player_embeddings.shape[0], self.rnn.hidden_size):
            h_0 = torch.zeros(
                self.rnn.num_layers, player_embeddings.shape[0], self.rnn.hidden_size, device=player_embeddings.device)
            c_0 = torch.zeros(
                self.rnn.num_layers, player_embeddings.shape[0], self.rnn.hidden_size, device=player_embeddings.device)
            self.prev_player_states = (h_0, c_0)
        if self.prev_inventory_states is None or self.prev_inventory_states[0].shape != (self.rnn.num_layers, inventory_embeddings.shape[0], self.rnn.hidden_size):
            h_0 = torch.zeros(
                self.rnn.num_layers, inventory_embeddings.shape[0], self.rnn.hidden_size, device=inventory_embeddings.device)
            c_0 = torch.zeros(
                self.rnn.num_layers, inventory_embeddings.shape[0], self.rnn.hidden_size, device=inventory_embeddings.device)
            self.prev_inventory_states = (h_0, c_0)
        if self.prev_hidden_states is None or self.prev_hidden_states[0].shape != (self.rnn.num_layers, hidden.shape[0], self.rnn.hidden_size):
            h_0 = torch.zeros(
                self.rnn.num_layers, hidden.shape[0], self.rnn.hidden_size, device=hidden.device)
            c_0 = torch.zeros(
                self.rnn.num_layers, hidden.shape[0], self.rnn.hidden_size, device=hidden.device)
            self.prev_hidden_states = (h_0, c_0)
        player_embeddings, self.prev_player_states = self.rnn(
            player_embeddings, self.prev_player_states)
        inventory_embeddings, self.prev_inventory_states = self.rnn(
            inventory_embeddings, self.prev_inventory_states)

        player_embeddings = self.attn(player_embeddings)
        inventory_embeddings = self.attn(inventory_embeddings)
        hidden, self.prev_hidden_states = self.rnn(
            hidden, self.prev_hidden_states)
        hidden = self.attn(hidden)
        hidden = hidden.squeeze(1)
        # Concat before and after attention and use MLP (advise communicaion) to get same shape as before
        player_embeddings = torch.cat(
            [player_embeddings_before, player_embeddings], dim=-1)
        inventory_embeddings = torch.cat(
            [inventory_embeddings_before, inventory_embeddings], dim=-1)
        hidden = torch.cat([hidden_before, hidden], dim=-1)

        player_embeddings = torch.nn.functional.relu(
            self.fc(player_embeddings), inplace=True)
        inventory_embeddings = torch.nn.functional.relu(
            self.fc(inventory_embeddings), inplace=True)
        hidden = torch.nn.functional.relu(self.fc(hidden), inplace=True)
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
        for key, layer in self.layers.items():
            mask = None
            mask = action_targets[key]
            embs = embeddings.get(key)
            if embs is not None and embs.shape[1] != mask.shape[1]:
                b, _, f = embs.shape
                zeros = torch.zeros(
                    [b, 1, f], dtype=embs.dtype, device=embs.device)
                embs = torch.cat([embs, zeros], dim=1)

            action = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action)
        # detach the prev states
        self.prev_player_states = (
            self.prev_player_states[0].detach(), self.prev_player_states[1].detach())
        self.prev_inventory_states = (
            self.prev_inventory_states[0].detach(), self.prev_inventory_states[1].detach())
        self.prev_hidden_states = (
            self.prev_hidden_states[0].detach(), self.prev_hidden_states[1].detach())
        return actions
