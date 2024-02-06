import os
import json
import shutil
import random
import logging
import argparse
from typing import Dict
from collections import defaultdict

import torch

import nmmo
import pufferlib
import pufferlib.emulation
import pufferlib.policy_store
import pufferlib.policy_ranker
from pufferlib.frameworks import cleanrl
from pufferlib.vectorization import Multiprocessing

from reinforcement_learning import clean_pufferl
import environment

VAR_TEST_NUM = 10  # use 10 different random seeds for the variability test

def get_eval_constants(download_dir="/home/aicrowd/",
                       test_num_episode=None,
                       use_gpu=True):
    return {
        "DEVICE": "cuda" if use_gpu else "cpu",
        "PVE_WORK_DIR": download_dir + "pve_eval",
        "SUBMISSION_PATH": download_dir + "submissions",
        "SUBMISSION_CONTENT": "my-submission",

        "EVAL_BATCH_SIZE": 2**15,
        "NUM_ENVS": test_num_episode or 8,

        # NOTE: when merging to the baselines repo, remove the pvp directory
        "SAMPLE_EVAL_TASK": "evaluation/sample_eval_task_with_embedding.pkl",  # 25 tasks
        "HELDOUT_EVAL_TASK": "evaluation/heldout_task_with_embedding_v4.pkl",  # 63 tasks

        "NUM_PVE_SAMPLE_EVAL_EPISODES": test_num_episode or 16,
        "NUM_PVE_HELDOUT_EVAL_EPISODES": test_num_episode or 32,
        "NUM_PVP_HELDOUT_EVAL_EPISODES": test_num_episode or 200,  # cannot do more due to memory leak
        "PVP_RESULT_PREFIX": "pvp_eval",
    }

def load_queue_file(queue_file):
    with open(queue_file, 'r') as f:
        queue = [sub_id.strip() for sub_id in f.readlines()]
    return queue

class EvalConfig(nmmo.config.Default):
    def __init__(self, task_file, mode):
        super().__init__()
        self.CURRICULUM_FILE_PATH = task_file

        # eval constants
        self.COMMUNICATION_SYSTEM_ENABLED = False
        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.PLAYER_N = 128
        self.HORIZON = 1024
        self.PLAYER_DEATH_FOG = None
        self.NPC_N = 128
        self.COMBAT_SPAWN_IMMUNITY = 20
        self.RESOURCE_RESILIENT_POPULATION = 0
        self.TASK_EMBED_DIM = 4096

        # map related
        self.TERRAIN_FLIP_SEED = True
        self.MAP_GENERATE_PREVIEWS = True
        self.MAP_FORCE_GENERATION = False
        self.MAP_CENTER = 128
        if mode not in ["pve", "pvp"]:
            raise ValueError(f"Unknown mode {mode}")
        if mode == "pve":
            self.MAP_N = 4
            self.PATH_MAPS = "maps/pve_eval/"
        else:
            self.MAP_N = 256
            self.PATH_MAPS = "maps/pvp_eval/"

def make_env_creator(task_file, mode):
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(EvalConfig(task_file, mode))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=environment.Postprocessor,
            postprocessor_kwargs={'eval_mode': True, 'early_stop_agent_num': 0,},
        )
        return env
    return env_creator

class AllPolicySelector(pufferlib.policy_ranker.PolicySelector):
    def select_policies(self, policies):
        policy_names = list(set(policies.keys()) - set(self._exclude_names))
        assert len(policy_names) == self._num, "Number of policies must match"
        policy_names.sort()
        return [policies[name] for name in policy_names]

class EvalRunner:
    def __init__(self, constants: Dict):
        self.policy_store_dir = None
        self.device = constants["DEVICE"]
        self.num_envs = constants["NUM_ENVS"]
        self.eval_batch_size = constants["EVAL_BATCH_SIZE"]
        self.sample_task_file = constants["SAMPLE_EVAL_TASK"]
        self.heldout_task_file = constants["HELDOUT_EVAL_TASK"]

    @staticmethod
    def create_policy_ranker(policy_store_dir, ranker_file="ranker.pickle", db_file="ranking.sqlite"):
        file = os.path.join(policy_store_dir, ranker_file)
        if os.path.exists(file):
            logging.info("Using existing policy ranker from %s", file)
            policy_ranker = pufferlib.policy_ranker.OpenSkillRanker.load_from_file(file)
        else:
            logging.info("Creating a new policy ranker and db under %s", policy_store_dir)
            db_file = os.path.join(policy_store_dir, db_file)
            policy_ranker = pufferlib.policy_ranker.OpenSkillRanker(db_file, "anchor")
        return policy_ranker

    def setup_evaluator(self, task_file, seed, mode):
        if not os.path.exists(self.policy_store_dir):
            raise ValueError("Policy store directory does not exist")
        policy_store = pufferlib.policy_store.DirectoryPolicyStore(self.policy_store_dir)
        policy_ranker = self.create_policy_ranker(self.policy_store_dir)
        num_policies = len(policy_store._all_policies())
        policy_selector = AllPolicySelector(num_policies)

        # Below is a dummy learner poliy
        from reinforcement_learning import policy  # import your policy
        def make_policy(envs):
            learner_policy = policy.Baseline(
                envs.driver_env,
                input_size=256,
                hidden_size=256,
                task_size=4096
            )
            return cleanrl.Policy(learner_policy)

        # Setup the evaluator. No training during evaluation
        evaluator = clean_pufferl.CleanPuffeRL(
            device=torch.device(self.device),
            seed=seed,
            env_creator=make_env_creator(task_file, mode),
            env_creator_kwargs={},
            agent_creator=make_policy,
            data_dir=self.policy_store_dir,
            vectorization=Multiprocessing,
            num_envs=self.num_envs,
            num_cores=self.num_envs,
            num_buffers=1,
            selfplay_learner_weight=0,
            selfplay_num_policies=num_policies + 1,
            batch_size=self.eval_batch_size,
            policy_store=policy_store,
            policy_ranker=policy_ranker,
            policy_selector=policy_selector,
        )
        return evaluator

    def perform_eval(self, task_file, seed, mode, num_eval_episodes, save_file_prefix):
        evaluator = self.setup_evaluator(task_file, seed, mode)
        #results = defaultdict(list)  # this is used for the aicrowd leaderboard
        eval_results = {}  # making a separate dict for the eval results
        cnt_episode = 0
        while cnt_episode < num_eval_episodes:
            _, _, infos = evaluator.evaluate()
            for pol, vals in infos.items():
                cnt_episode += sum(infos[pol]["episode_done"])
                # results for the aicrowd
                #results[pol].extend([e[1] for e in infos[pol]['team_results']])
                if pol not in eval_results:
                    eval_results[pol] = defaultdict(list)
                for k, v in vals.items():
                    if k == "length":
                        eval_results[pol][k] += v  # length is a plain list
                    if k.startswith("curriculum"):
                        eval_results[pol][k] += [vv[0] for vv in v]
            print(f"Evaluated {cnt_episode} episodes.\n")
        self._save_results(eval_results, f"{save_file_prefix}_{seed}.json")
        evaluator.close()
        return eval_results

    def _save_results(self, results, file_name):
        with open(os.path.join(self.policy_store_dir, file_name), "w") as f:
            json.dump(results, f)

        # # Save the result file (for the aicrowd leaderboard)
        # result_dir = save_result_dir
        # print( result_dir )
        # os.system(f"mkdir -p {result_dir}")
        # result_path = os.path.join(result_dir, f"result-{Constants.NMMO_ROLLOUT_NAME}-{time.strftime('%Y%m%d_%H%M%S')+'.pkl'}")
        # print( result_path )
        # logger.info("dump result")
        # util.write_data(
        #     pickle.dumps(results),
        #     result_path,
        #     binary=True,
        # )
        # logger.info("dump result done")

    def run(self, seed):
        raise NotImplementedError

class FirstRoundEvalRunner(EvalRunner):  # PvE mode
    def __init__(self, submission_id, constants: Dict):
        super().__init__(constants)
        self.submission_id = submission_id
        self.num_sample_eval_episodes = constants["NUM_PVE_SAMPLE_EVAL_EPISODES"]
        self.num_heldout_eval_episodes = constants["NUM_PVE_HELDOUT_EVAL_EPISODES"]
        # verify submission and create the work dir
        logging.info(f"Preparing workdir for {self.submission_id}")
        if not self._prepare_workdir(constants):
            raise RuntimeError("Failed to prepare workdir")

    def _prepare_workdir(self, constants):
        # see if the submission directory exists
        submission = os.path.join(constants["SUBMISSION_PATH"], self.submission_id)
        assert os.path.exists(submission), f"{submission} does not exist"
        
        # see if the pt or pkl file exists
        sub_folder = os.path.join(submission, constants["SUBMISSION_CONTENT"])
        pt_files = [f for f in os.listdir(sub_folder) if f.endswith('.pt')]
        pkl_files = [f for f in os.listdir(sub_folder) if f.endswith('.pkl')]
        assert len(pt_files) == 1 or len(pkl_files) == 1, "Submission must contain exactly one .pt or .pkl file"

        # create the work dir
        work_dir = os.path.join(constants["PVE_WORK_DIR"], self.submission_id)
        os.makedirs(work_dir, exist_ok=True)
        if len(pt_files) == 1:
            shutil.copy(os.path.join(sub_folder, pt_files[0]), work_dir)
        if len(pkl_files) == 1:
            shutil.copy(os.path.join(sub_folder, pkl_files[0]), work_dir)
        self.policy_store_dir = work_dir
        return True

    def run(self, seed):
        assert self.policy_store_dir is not None, "Policy store directory must be specified"
        logging.info(f"Evaluating {self.submission_id} in the PvE mode (the first round) with {seed}")
        # run the sample tasks
        self.perform_eval(self.sample_task_file, seed, "pve", self.num_sample_eval_episodes,
                          save_file_prefix=f"eval_sample_{self.submission_id}")
        # run the heldout tasks
        self.perform_eval(self.heldout_task_file, seed, "pve", self.num_heldout_eval_episodes,
                          save_file_prefix=f"eval_heldout_{self.submission_id}")

class SecondRoundEvalRunner(EvalRunner):  # PvP mode
    def __init__(self, policy_store_dir, constants: Dict, run_pve=False):
        super().__init__(constants)
        self.run_pve = run_pve
        self.num_sample_eval_episodes = constants["NUM_PVE_SAMPLE_EVAL_EPISODES"]
        self.num_heldout_pve_episodes = constants["NUM_PVE_HELDOUT_EVAL_EPISODES"]
        self.num_heldout_pvp_episodes = constants["NUM_PVP_HELDOUT_EVAL_EPISODES"]
        self.policy_store_dir = policy_store_dir
        self.result_file_prefix = constants["PVP_RESULT_PREFIX"]
        if not os.path.exists(self.policy_store_dir):
            raise RuntimeError(f"{self.policy_store_dir} does not exist")

    def run(self, seed):
        assert self.policy_store_dir is not None, "Policy store directory must be specified"
        if not self.run_pve:
            logging.info(f"Evaluating {self.policy_store_dir} in the PvP mode (the second round) with {seed}")
            logging.info(f"Using the task file: {self.heldout_task_file}")
            self.perform_eval(self.heldout_task_file, seed, "pvp", self.num_heldout_pvp_episodes,
                            save_file_prefix=self.result_file_prefix)
        else:
            # NOTE: this assumes there is only one policy in the policy store, but does not explicitly check
            assert self.policy_store_dir is not None, "Policy store directory must be specified"
            logging.info(f"Evaluating {self.policy_store_dir} in the PvE mode (the first round) with {seed}")
            # run the sample tasks
            self.perform_eval(self.sample_task_file, seed, "pve", self.num_sample_eval_episodes,
                            save_file_prefix="eval_sample_pve")
            # run the heldout tasks
            self.perform_eval(self.heldout_task_file, seed, "pve", self.num_heldout_pve_episodes,
                            save_file_prefix="eval_heldout_pve")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--submission-id",
        dest = "submission_id",
        type = str,
        default = "241095",  # baselines submission id
        help = "Valid submission id in the submissions directory. If not provided, the queue file must be provided.",
    )
    parser.add_argument(
        "-q",
        "--queue-file",
        dest = "queue_file",
        type = str,
        default = None,
        help = "Path to the queue file. If not provided, the submission id must be provided.",
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        dest = "random_seed",
        type = int,
        default = 14757270,
        help = "Random seed for the evaluation",
    )
    parser.add_argument(
        "-t",
        "--test-var",
        dest = "test_var",
        action = "store_true",
        help = "Test the variability of the metrics using the baseline policy and different random seeds",
    )
    parser.add_argument(
        "-p",
        "--policy-store-dir",
        dest = "policy_store_dir",
        type = str,
        default = None,
        help = "Path to the policy store directory for the PvP evaluation",
    )
    parser.add_argument(
        "-e",
        "--run-pve-eval",
        dest = "run_pve_eval",
        action = "store_true",
        help = "Run the PvE evaluation for the specified policy store directory",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest = "debug",
        action = "store_true",
        help = "Run the evaluation in the debug mode",
    )
    # Parse and check the arguments
    pve_args = parser.parse_args()
    assert pve_args.submission_id is not None or pve_args.queue_file is not None \
        or pve_args.policy_store_dir is not None,\
        "Please provide either submission id, the queue file, or the policy store directory"

    eval_constants = get_eval_constants(test_num_episode=4 if pve_args.debug else None)

    if pve_args.policy_store_dir is not None:
        assert os.path.exists(pve_args.policy_store_dir),\
            f"{pve_args.policy_store_dir} does not exist"
        run_pve = pve_args.run_pve_eval or False
        eval_runner = SecondRoundEvalRunner(pve_args.policy_store_dir, eval_constants, run_pve)
        eval_runner.run(pve_args.random_seed)

    elif pve_args.queue_file is not None:
        processed = []
        queue = load_queue_file(pve_args.queue_file)
        while len(queue) > 0:
            sub_id = queue.pop(0).strip()
            # process sub_id
            eval_runner = FirstRoundEvalRunner(sub_id, eval_constants)
            eval_runner.run(pve_args.random_seed)

            # add to processed
            processed.append(sub_id)

            # open the queue file to see if there is any new submission
            tmp_queue = load_queue_file(pve_args.queue_file)
            for sub in tmp_queue:
                if sub not in processed and sub not in queue:
                    queue.append(sub)

    elif getattr(pve_args, "test_var", False):
        # Run the evaluation with different seeds on the same submission
        # If one is using just -t, then the baseline submission id is used
        eval_constants["WORK_DIR"] = "/home/aicrowd/pve_var_test/"
        os.makedirs(eval_constants["WORK_DIR"], exist_ok=True)
        eval_runner = FirstRoundEvalRunner(pve_args.submission_id, eval_constants)
        for _ in range(VAR_TEST_NUM):
            seed = random.randint(10000000, 99999999)
            eval_runner.run(seed)

    else:
        # process single submission
        eval_runner = FirstRoundEvalRunner(pve_args.submission_id, eval_constants)
        eval_runner.run(pve_args.random_seed)
