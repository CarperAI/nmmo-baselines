import os
import time
import pickle
import logging
from types import SimpleNamespace
from typing import Callable

import torch
from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl

BASELINE_CURRICULUM_FILE = "reinforcement_learning/curriculum_with_embedding.pkl"

# args to override
CONST_ARGS = {
    "device": "cuda",
    "num_envs": 6,
    "num_buffers": 2,
    "train_num_steps": 30_000_000,  # training will stop early
    "checkpoint_interval": 100,
    "runs_dir": "/tmp/runs",
    "wandb_entity": "kywch",
    "wandb_project":"nmmo-contrib",
    # NOTE: check if different settings for maps were used
    "maps_path": "maps/train/",
    "map_size": 128,
    "num_maps": 1024,  # top submissions used larger number of maps
}


def get_config_args(config_module, curriculum_file=None, debug=False, cli_args=False):
    args = SimpleNamespace(**config_module.Config.asdict())
    for k, v in CONST_ARGS.items():
        setattr(args, k, v)
    if curriculum_file:
        args.tasks_path = curriculum_file
    if cli_args:
        # This overrides all the above, and can get the args from the command line
        args = config_module.create_config(config_module.Config)
    if debug:
        args.num_envs = 1
        args.num_buffers = 1
        args.use_serial_vecenv = True
        args.rollout_batch_size = 2**10
        args.wandb_entity = None
        args.wandb_project = None
    return args

class TrainHelper:
    def __init__(self,
                 run_prefix,
                 args,
                 env_creator: Callable,
                 agent_creator: Callable,
                 pufferl_cls,
                 policy_src,
                 train_kwargs = None):
        self.run_prefix = run_prefix
        self.args = args
        self.env_creator = env_creator
        self.agent_creator = agent_creator
        self.pufferl_cls = pufferl_cls
        self.policy_src = policy_src
        self.train_kwargs = train_kwargs
        if self.train_kwargs is None:
            self.train_kwargs = {  # baselines train kwargs
                "update_epochs": args.ppo_update_epochs,
                "bptt_horizon": args.bptt_horizon,
                "batch_rows": args.ppo_training_batch_size // args.bptt_horizon,
                "clip_coef": args.clip_coef,
            }

    def _make_trainer(self, seed=None):
        args = self.args
        seed = seed or args.seed
        # create a time-stamped run name when instantiating a new trainer
        args.run_name = f"{self.run_prefix}_{seed}_{int(time.time()) % 10_000_000}"
        self.run_dir = os.path.join(args.runs_dir, args.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        logging.info("Training run: %s (%s)", args.run_name, self.run_dir)
        logging.info("Training args: %s", args)

        policy_store = None
        if args.policy_store_dir is None:
            args.policy_store_dir = os.path.join(self.run_dir, "policy_store")
            logging.info("Using policy store from %s", args.policy_store_dir)
            policy_store = DirectoryPolicyStore(args.policy_store_dir)

        return self.pufferl_cls(
            device=torch.device(args.device),
            seed=seed,
            env_creator=self.env_creator(args),
            env_creator_kwargs={},
            agent_creator=self.agent_creator,
            data_dir=self.run_dir,
            exp_name=args.run_name,
            policy_store=policy_store,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_extra_data=args,
            checkpoint_interval=args.checkpoint_interval,
            vectorization=Serial if args.use_serial_vecenv else Multiprocessing,
            total_timesteps=args.train_num_steps,
            num_envs=args.num_envs,
            num_cores=args.num_cores or args.num_envs,
            num_buffers=args.num_buffers,
            batch_size=args.rollout_batch_size,
            learning_rate=args.ppo_learning_rate,
            selfplay_learner_weight=args.learner_weight,
            selfplay_num_policies=args.max_opponent_policies + 1,
        )

    def _train(self, trainer):
        """Override this function to provide custom args for pufferl.train"""
        trainer.train(
            update_epochs=self.args.ppo_update_epochs,
            bptt_horizon=self.args.bptt_horizon,
            batch_rows=self.args.ppo_training_batch_size // self.args.bptt_horizon,
            clip_coef=self.args.clip_coef,
        )

    def _save_final_policy(self, trainer):
        trainer._save_checkpoint()
        policies = trainer.policy_store._all_policies()
        keys = list(policies.keys())
        keys.sort()
        final_policy = policies[keys[-1]].policy().to("cpu")  # the last checkpoint
        checkpoint = {
            "policy_src": self.policy_src,
            "state_dict": final_policy.state_dict(),
        }
        out_name = f"{self.run_dir}_{trainer.update}.pkl"
        with open(out_name, "wb") as out_file:
            pickle.dump(checkpoint, out_file)

    def run(self, seed=None, time_limit_sec=8 * 3600):  # 8 hours
        trainer = self._make_trainer(seed)
        start_time = time.time()
        while not trainer.done_training():
            trainer.evaluate()
            trainer.train(**self.train_kwargs)
            if (time.time() - start_time) > time_limit_sec:
                break
        self._save_final_policy(trainer)
        trainer.close()

def get_train_helper_baseline(debug=False):
    run_prefix = "baseline"
    import environment
    from reinforcement_learning import clean_pufferl, policy, config

    args = get_config_args(config, BASELINE_CURRICULUM_FILE, debug)

    def make_policy(envs):
        learner_policy = policy.Baseline(
            envs.driver_env,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            task_size=args.task_size
        )
        return cleanrl.Policy(learner_policy)

    policy_file = "reinforcement_learning/policy.py"
    with open(policy_file, "r") as f:
        policy_src = f.read()

    policy_src += f"""

class Config(nmmo.config.Default):
    PROVIDE_ACTION_TARGETS = True
    PROVIDE_NOOP_ACTION_TARGET = True
    MAP_FORCE_GENERATION = False
    TASK_EMBED_DIM = 4096
    COMMUNICATION_SYSTEM_ENABLED = False

def make_policy():
    from pufferlib.frameworks import cleanrl
    env = pufferlib.emulation.PettingZooPufferEnv(nmmo.Env(Config()))
    # Parameters to your model should match your configuration
    learner_policy = Baseline(
        env,
        input_size={args.input_size},
        hidden_size={args.input_size},
        task_size=4096
    )
    return cleanrl.Policy(learner_policy)

"""

    return TrainHelper(run_prefix, args,
                       environment.make_env_creator,
                       make_policy,
                       clean_pufferl.CleanPuffeRL,
                       policy_src)
