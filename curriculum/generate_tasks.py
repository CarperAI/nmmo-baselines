import argparse

from nmmo.task.base_predicates import *
from nmmo.task.group import Group, complement

import nmmo
from nmmo.io import action as Action
from nmmo.lib.material import All as Materials
from nmmo.core.env import Env as TaskEnv
from nmmo.task.utils import TeamHelper
from nmmo.task.task_api import Once
from nmmo.task.predicate import AND, OR, NOT

import task_generator
from scripted import baselines

"""
Script to generate and save baseline tasks for 2023 competition.
The default args in this script are the ones used to generate the competition baseline tasks.
"""

# TODO type hints
# TODO set seeds
# TODO add command line args for generate_task_definitions params
# TODO add more base_predicates (when RandomTaskGenerator supports);
#      want to include any CountEvent?

def generate_task_definitions(n: int=1,
                   max_clauses: int=4,
                   max_clause_size: int=3,
                   not_p: float=0.5,
                   num_rows: int = 1024,
                   num_cols: int = 1024,):
    """
    Generates Task definitions by heuristic input/output shaping with RandomTaskGenerator.


    Args:
        max_clauses: max clauses in each Task
        max_clause_size: max Predicates in each clause
        not_p: probability that a Predicate will be NOT'd
        num_rows: num rows in map
        num_cols: num cols in map

    """

    # Predicate specific max values for rng

    # level for any skill to require for any task
    max_skill_level: int = 15
    # number of hits scored in combat style
    max_score_hit_n: int = 100
    # amount of gold hoarded
    max_hoard_gold_amount: int = 100
    # amount of gold earned
    max_earn_gold_amount: int = 100
    # amount of gold spent
    max_spend_gold_amount: int = 100
    # amount of profit
    max_profit_amount: int = 100
    # number of agents per team
    agents_per_team = 16


    gen = task_generator.RandomTaskGenerator()
    # gen.add_pred_spec(CanSeeTile, [])
    # gen.add_pred_spec(StayAlive, [])
    # gen.add_pred_spec(AllDead, [])
    gen.add_pred_spec(OccupyTile, [[i for i in range(num_rows)],
                                   [i for i in range(num_cols)]])
    gen.add_pred_spec(AllMembersWithinRange, [[i for i in range(max(num_rows,num_cols))]])
    # gen.add_pred_spec(CanSeeAgent, [])
    # gen.add_pred_spec(CanSeeGroup, [])
    gen.add_pred_spec(DistanceTraveled, [[i for i in range(max(num_rows//2,num_cols//2))]])
    # gen.add_pred_spec(AttainSkill, [[skills],
    #                                 [i for i in range(1, max_skill_level+1)],
    #                                 [i for i in range(1, agents_per_team+1)]])
    # gen.add_pred_spec(ScoreHit, [[combat_skills],
    #                              [i for i in range(max_score_hit_n)]])
    gen.add_pred_spec(HoardGold, [[i for i in range(1, max_hoard_gold_amount)]])
    gen.add_pred_spec(EarnGold, [[i for i in range(1, max_earn_gold_amount)]])
    gen.add_pred_spec(SpendGold, [[i for i in range(1, max_spend_gold_amount)]])
    gen.add_pred_spec(MakeProfit, [[i for i in range(1, max_profit_amount)]])

    task_infos = []
    for _ in range(n):
        task_infos.append(gen.sample(max_clauses=max_clauses,
                              max_clause_size=max_clause_size,
                              not_p=not_p))
    return task_infos

# tmp test config
class ScriptedAgentTestConfig(nmmo.config.Small, nmmo.config.AllGameSystems):
  __test__ = False

  LOG_ENV = True

  LOG_MILESTONES = True
  LOG_EVENTS = False
  LOG_VERBOSE = False

  SPECIALIZE = True
  PLAYERS = [
    baselines.Fisher, baselines.Herbalist,
    baselines.Prospector,baselines.Carver, baselines.Alchemist,
    baselines.Melee, baselines.Range, baselines.Mage]

def run_tasks(task_infos):
    config = ScriptedAgentTestConfig()
    env = TaskEnv(config)
    team_helper = TeamHelper.generate_from_config(config)

    def task_assigner(team_helper, task_info):
        """
        Takes the generated clause info, instantiates the Predicates, and combines them all using 
        Conjunctive Normal Form (CNF).
        """
        task_assignments = []
        for team in team_helper.all_teams():
            clauses = []
            for clause_info in task_info:
                predicates = []
                for pred_info in clause_info:
                    # instantiate Predicate
                    pred_class = pred_info[0]
                    predicate = pred_class(team, *pred_info[2:]) # [2:] contains the params for the pred_class
                    if pred_info[1]: predicate = NOT(predicate)
                    predicates.append(predicate)

                clause = OR(*predicates)
                clauses.append(clause)
            
            clauses = AND(*clauses)

            task_assignments.append(Once(team, clauses))

        return task_assignments
    
    # Test rollout
    for task_info in task_infos:
        task_assignments = task_assigner(team_helper, task_info)
        env.change_task(task_assignments)
        for _ in range(20):
            env.step({})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n', default=10, help='number of tasks to generate')
    args = parser.parse_args()

    task_infos = generate_task_definitions(n=int(args.n))

    # TODO: implement save and load generated task_infos
    # for now just continue to run_tasks with `task_infos` in memory
    for t in task_infos:
        print('TASK:', t)
        print()

    # test run tasks
    # TODO: probably expand and move this stuff to another file for demo
    run_tasks(task_infos)
