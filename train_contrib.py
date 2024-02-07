import logging
import argparse
import importlib

import contrib
import train_helper

def get_train_helper(policy_name, debug=False):
    if policy_name == "baseline":
        return train_helper.get_train_helper_baseline(debug)
    elif policy_name in contrib.TESTED:
        submission = importlib.import_module(f"contrib.sub_{policy_name}")
    else:
        raise ValueError(f"Unknown policy name: {policy_name}")
    return submission.get_train_helper(debug)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy",
        dest = "policy_to_train",
        type = str,
        nargs = "+",
        default = ["baseline"],
        help = "A list of policies to train (Default: baseline)",
    )
    parser.add_argument(
        "-a",
        "--train_all",
        dest = "train_all",
        action = "store_true",
        help = "Train all models (Default: False)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest = "seed",
        type = int,
        default = None,
        help = "Random integer seed (Default: None, which uses the seed specified in the config file)",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        dest = "time_limit_sec",
        type = int,
        default = 8 * 3600,
        help = "Training time limit in seconds (Default: 8 hours)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest = "debug_mode",
        action = "store_true",
        help = "Debug mode (Default: False)",
    )
    parser.add_argument(
        "-i",
        "--run-identifier",
        dest = "run_identifier",
        type = str,
        default = None,
        help = "Pre-fix to attach to run names (Default: None)",
    )

    args = parser.parse_args()
    debug_flag = False
    time_limit_sec = args.time_limit_sec
    if args.debug_mode:
        debug_flag = True
        time_limit_sec = 30
        logging.info("Running in debug mode")

    if len(args.policy_to_train) == 0 and not args.train_all:
        raise ValueError("No policies to train")
    policy_to_train = args.policy_to_train

    # Check if MAX_NUM_MAPS (1024) maps are available, and generate these if not
    train_helper.check_maps()

    if args.train_all:
        policy_to_train = contrib.TESTED + ["baseline"]

    for pol in policy_to_train:
        train_helper = get_train_helper(pol, debug=debug_flag)
        if args.run_identifier:
            train_helper.run_prefix = args.run_identifier + "_" + train_helper.run_prefix
        train_helper.run(args.seed, time_limit_sec)
