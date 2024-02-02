from train_helper import TrainHelper, get_config_args, BASELINE_CURRICULUM_FILE
from pufferlib.frameworks import cleanrl

SUBMISSION_ID = "246748"

def get_train_helper(debug=False):
    run_prefix = f"s{SUBMISSION_ID}"
    from . import environment, config, policy
    from reinforcement_learning import clean_pufferl

    args = get_config_args(config, BASELINE_CURRICULUM_FILE, debug)

    def make_policy(envs):
        learner_policy = policy.ReducedModelV2(
            envs.driver_env,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            task_size=args.task_size
        )
        return cleanrl.Policy(learner_policy)

    policy_file = f"contrib/sub_{SUBMISSION_ID}/policy.py"
    with open(policy_file, "r") as f:
        policy_src = f.read()

    # NOTE: For the PvP eval to work correctly, any custom components, like network blocks, etc,
    #       must be included in the policy file.
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
    learner_policy = ReducedModelV2(
        env,
        input_size={args.input_size},
        hidden_size={args.input_size},
        task_size=4096
    )
    return cleanrl.Policy(learner_policy)

"""

    train_kwargs = {
        "update_epochs": args.ppo_update_epochs,
        "bptt_horizon": args.bptt_horizon,
        "batch_rows": args.ppo_training_batch_size // args.bptt_horizon,
        "clip_coef": args.clip_coef,
        "clip_vloss": not args.no_clip_vloss,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
    }

    return TrainHelper(run_prefix, args,
                       environment.make_env_creator,
                       make_policy,
                       clean_pufferl.CleanPuffeRL,
                       policy_src,
                       train_kwargs)
