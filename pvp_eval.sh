#!/bin/bash

# Get command line arguments
policy_store_dir=$1
num_repeats=$2

# Check if policy_store_dir and num_repeats are provided
if [ -z "$policy_store_dir" ] || [ -z "$num_repeats" ]
then
  echo "Usage: $0 <policy_store_dir> <num_repeats>"
  exit 1
fi

# Run eval_runner only once
if [ "$num_repeats" -eq 1 ]
then
  echo "Running pvp eval once with the default seed"
  python -u eval_runner.py -p $policy_store_dir
  exit 0
fi

# Run eval_runner.py num_repeats times
for (( i=1; i<=$num_repeats; i++ ))
do
  seed=$(($RANDOM * 1000 + 10000000 + $RANDOM))
  echo "Running pvp eval with seed $seed"
  python -u run_eval.py -r $seed -p $policy_store_dir
done
