import os
import json
import argparse

import numpy as np
import pandas as pd

# string matching for task names
WEIGHT_DICT = {
    "TickGE": ("survival", 100 / 6),  # 1 survival task
    "PLAYER_KILL": ("combat", 100 / (6*3)),  # 3 combat tasks
    "DefeatEntity": ("combat", 100 / (6*3)),
    "GO_FARTHEST": ("exploration", 100 / (6*2)),  # 2 exploration tasks
    "OccupyTile": ("exploration", 100 / (6*2)),
    "AttainSkill": ("skill", 100 / (6*8)),  # 8 skill tasks
    "HarvestItem": ("item", 100 / (6*44)),  # 44 item tasks
    "ConsumeItem": ("item", 100 / (6*44)),
    "EquipItem": ("item", 100 / (6*44)),
    "FullyArmed": ("item", 100 / (6*44)),
    "EARN_GOLD": ("market", 100 / (6*5)),  # 5 market tasks
    "BUY_ITEM": ("market", 100 / (6*5)),
    "EarnGold": ("market", 100 / (6*5)),
    "HoardGold": ("market", 100 / (6*5)),
    "MakeProfit": ("market", 100 / (6*5)),
}

def get_task_weight(task_name):
    for key, val in WEIGHT_DICT.items():
        if key in task_name:
            return val
    return None, 0
    #raise ValueError(f"Task name {task_name} not found in weight dict")

def summarize_single_eval(data, weighted_score=False):
    # get the mean, and quantile values for each
    # if the key starts with curriculum, then also get the proportion of vals >= 1
    summary = {}
    # task-level info
    for key, vals in data.items():
        summary[key] = {"count": len(vals), "mean": np.mean(vals), "median": np.median(vals)}
        if key.startswith("curriculum"):
            completed = [1 if v >= 1 else 0 for v in vals]
            over30pcnt = [1 if v >= 0.3 else 0 for v in vals]
        if key == "length":
            completed = [1 if v >= 1023 else 0 for v in vals]  # full episode length
            over30pcnt = [1 if v >= 300 else 0 for v in vals]
        summary[key]["completed"] = np.mean(completed) if len(completed) > 0 else 0
        summary[key]["over30pcnt"] = np.mean(over30pcnt) if len(over30pcnt) > 0 else 0
        if weighted_score and key != "length":
            category, weight = get_task_weight(key)
            summary[key]["category"] = category
            summary[key]["weight"] = weight
            summary[key]["weighted_score"] = summary[key]["completed"] * weight
    # meta info
    summary["avg_progress"] = np.mean([v["mean"] for k, v in summary.items()
                                       if k.startswith("curriculum")])
    if weighted_score:
        summary["weighted_score"] = np.sum([v["weighted_score"] for k, v in summary.items()
                                            if k.startswith("curriculum")])
    return summary

def process_pve_score(work_dir,
                      sample_eval_prefix="eval_sample_",
                      heldout_eval_prefix="eval_heldout_"):
    eval_files = os.listdir(work_dir)
    # NOTE: only one sample and heldout eval results file is expected
    sample_eval_file = [f for f in eval_files if f.startswith(sample_eval_prefix) and f.endswith(".json")][0]
    heldout_eval_file = [f for f in eval_files if f.startswith(heldout_eval_prefix) and f.endswith(".json")][0]
    assert sample_eval_file and heldout_eval_file, "No eval results files found"

    with open(os.path.join(work_dir, sample_eval_file), "r") as f:
        data = json.load(f)
        data = list(data.values())[0]
        sample_eval_summary = summarize_single_eval(data)
    with open(os.path.join(work_dir, heldout_eval_file), "r") as f:
        data = json.load(f)
        data = list(data.values())[0]
        heldout_eval_summary = summarize_single_eval(data, weighted_score=True)

    results = {}
    results["sample_eval_length"] = sample_eval_summary["length"]["mean"]
    results["sample_eval_score"] = sample_eval_summary["avg_progress"]
    results["heldout_eval_length"] = heldout_eval_summary["length"]["mean"]
    results["heldout_eval_score"] = heldout_eval_summary["avg_progress"]
    results["heldout_weighted_score"] = heldout_eval_summary["weighted_score"]
    print("PvE eval results:", ", ".join(map(str, list(results.values()))))
    return results

def process_batch_pve_score(batch_dir):
    # NOTE: assume that the batch_dir contains subdirectories for each submission
    sub_dirs = [d for d in os.listdir(batch_dir)]
    eval_summary = []
    for sub in sub_dirs:
        sub_path = os.path.join(batch_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        eval_files = os.listdir(sub_path)
        if len([f for f in eval_files if f.endswith(".json")]) < 2:  # insufficient eval results
            continue
        policy_file = [f for f in eval_files if f.endswith(".pt") or f.endswith(".pkl")][0]
        sub_results = {
            "submission_id": sub,
            "policy_file": policy_file,
            "policy_size": os.path.getsize(os.path.join(sub_path, policy_file)),
        }
        sub_results.update(process_pve_score(sub_path))
        eval_summary.append(sub_results)

    # save the summary results to csv
    df = pd.DataFrame(eval_summary)
    df.sort_values(by=["heldout_weighted_score"], inplace=True, ascending=False)
    df.to_csv(os.path.join(batch_dir, "submission_results.csv"), index=False)
    return df

def process_pve_var_test(work_dir, eval_prefix="eval_"):
    results = []
    for file in os.listdir(work_dir):
        # NOTE: assumes the file naming convention is "eval_<type>_<policy>_<seed>.json"
        if not file.startswith(eval_prefix) or not file.endswith(".json"):
            continue
        single_results = {}
        single_results["type"] = file.split("_")[1]
        single_results["seed"] = file.split("_")[3].replace(".json", "")
        with open(os.path.join(work_dir, file), "r") as f:
            data = json.load(f)
            data = list(data.values())[0]
        get_weighted = "heldout" in single_results["type"]
        summary = summarize_single_eval(data, weighted_score=get_weighted)
        single_results["length"] = summary["length"]["mean"]
        single_results["score"] = summary["avg_progress"]
        if get_weighted:
            single_results["weighted_score"] = summary["weighted_score"]
        results.append(single_results)

    df = pd.DataFrame(results)
    df.sort_values(by=["type", "length"], inplace=True)
    print(df)
    df.to_csv(os.path.join(work_dir, "eval_var_test.csv"), index=False)
    return df

def process_pvp_score(work_dir, eval_prefix="pvp_eval"):
    summ_policy = []
    summ_task = []
    for file in os.listdir(work_dir):
        # NOTE: assumes the file naming convention is "pvp_eval<type>_<seed>.json"
        if not file.startswith(eval_prefix) or not file.endswith(".json"):
            continue
        eval_type = file.split("_")[1]
        random_seed = file.split("_")[2].replace(".json", "")
        with open(os.path.join(work_dir, file), "r") as f:
            data = json.load(f)
        for pol_id, pol_data in data.items():
            summary = summarize_single_eval(pol_data, weighted_score=True)
            summ_policy.append({
                "submission_id": pol_id,
                "eval_type": eval_type,
                "seed": random_seed,
                "count": summary["length"]["count"],
                "length": summary["length"]["mean"],
                "score": summary["avg_progress"],
                "weighted_score": summary["weighted_score"]
            })

            # also gather the results across random seeds for each task, then average
            for task_name, task_data in summary.items():
                if not task_name.startswith("curriculum"):
                    continue
                summ_task.append({
                    "category": task_data["category"],
                    "task_name": task_name,
                    "weight": task_data["weight"],
                    "submission_id": pol_id,
                    "eval_type": eval_type,
                    "seed": random_seed,
                    "count": task_data["count"],
                    "score": task_data["mean"]
                })

    summ_df = pd.DataFrame(summ_policy)
    summ_df.sort_values(by=["submission_id", "eval_type", "seed"], inplace=True)
    summ_df.to_csv(os.path.join(work_dir, "pvp_score_by_seed.csv"), index=False)
    summ_grp = summ_df.groupby(["submission_id", "eval_type"])[["score", "weighted_score"]].mean().reset_index()
    summ_grp.to_csv(os.path.join(work_dir, "pvp_score_summary.csv"), index=False)
    print(summ_grp)

    task_df = pd.DataFrame(summ_task)
    task_df.sort_values(by=["category", "task_name", "submission_id", "eval_type", "seed"], inplace=True)
    task_df.to_csv(os.path.join(work_dir, "pvp_score_by_task_seed.csv"), index=False)
    task_grp = task_df.groupby(["category", "task_name", "submission_id", "eval_type"])[["score"]].mean().reset_index()
    task_grp.to_csv(os.path.join(work_dir, "pvp_score_task_summary.csv"), index=False)
    cate_grp = task_df.groupby(["category", "submission_id", "eval_type"])[["score"]].mean().reset_index()
    cate_grp.to_csv(os.path.join(work_dir, "pvp_score_category_summary.csv"), index=False)
    print(task_grp)
    return summ_df, summ_grp, task_df, task_grp, cate_grp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--pve-dir",
        dest = "pve_dir",
        type = str,
        default = None,
        help = "Path to the directory containing the PvE eval result json files"
    )
    parser.add_argument(
        "-t",
        "--pve-test-dir",
        dest = "pve_test_dir",
        type = str,
        default = None,
        help = "Path to the directory containing the PvE eval var test json files"
    )
    parser.add_argument(
        "-p",
        "--pvp-dir",
        dest = "pvp_dir",
        type = str,
        default = None,
        help = "Path to the directory containing the PvP eval result json files"
    )
    parser.add_argument(
        "--batch-pve-dir",
        dest = "batch_pve_dir",
        type = str,
        default = None,
        help = "Path to the directory containing multiple PvE eval result json files"
    )
    score_args = parser.parse_args()
    assert score_args.pve_dir or score_args.pve_test_dir or score_args.pvp_dir or score_args.batch_pve_dir,\
        "At least one of the PvE or PvP directories must be provided"

    if score_args.pve_dir:
        process_pve_score(score_args.pve_dir)
    if score_args.pve_test_dir:
        process_pve_var_test(score_args.pve_test_dir)
    if score_args.pvp_dir:
        process_pvp_score(score_args.pvp_dir)
    if score_args.batch_pve_dir:
        process_batch_pve_score(score_args.batch_pve_dir)
