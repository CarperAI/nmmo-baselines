import logging
import argparse
from train_helper import get_train_helper_baseline

def get_train_helper(policy_name, debug=False):
    if policy_name == "baseline":
        return get_train_helper_baseline(debug)
    elif policy_name == "246505":
        import contrib.sub_246505 as submission
    elif policy_name == "246539":
        import contrib.sub_246539 as submission
    elif policy_name == "246748":
        import contrib.sub_246748 as submission
    else:
        raise ValueError(f"Unknown policy name: {policy_name}")
    return submission.get_train_helper(debug)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy",
        dest="policy_to_train",
        type=str,
        default="baseline",
        help="Policy to train (Default: baseline)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Random integer seed (Default: None, which uses the seed specified in the config file)",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        dest="time_limit_sec",
        type=int,
        default=8 * 3600,
        help="Training time limit in seconds (Default: 8 hours)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug_mode",
        action="store_true",
        help="Debug mode (Default: False)",
    )

    args = parser.parse_args()
    debug_flag = False
    time_limit_sec = args.time_limit_sec
    if args.debug_mode:
        debug_flag = True
        time_limit_sec = 30
        logging.info("Running in debug mode")

    train_helper = get_train_helper(args.policy_to_train, debug=debug_flag)
    train_helper.run(args.seed, time_limit_sec)
