#!/bin/bash

python -u -O -m tools.evaluate \
--model.policy_pool=/fsx/home-daveey/experiments/pool.json \
--env.num_maps=100 \
--env.num_npcs=256 \
--env.death_fog_tick=100 \
--eval.num_rounds=1000000 \
--eval.num_policies=8 \
"${@}"
