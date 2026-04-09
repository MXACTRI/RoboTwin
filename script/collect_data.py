import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import math
import pickle
import numpy as np
from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _flatten_joint_positions(path_list):
    position_seq = []
    for result in path_list:
        if not isinstance(result, dict):
            continue
        if result.get("status") != "Success":
            continue
        pos = result.get("position")
        if pos is None:
            continue
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[None, :]
        if pos.shape[0] == 0:
            continue
        position_seq.append(pos)
    if len(position_seq) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.concatenate(position_seq, axis=0)


def _calc_path_length(positions):
    if positions.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _calc_smoothness(positions):
    if positions.shape[0] < 3:
        return 0.0
    accel = np.diff(positions, n=2, axis=0)
    return float(np.linalg.norm(accel, axis=1).mean())


def _calc_ee_path_length(path_list):
    """
    优先使用轨迹中可能存在的末端执行器位姿字段；若不存在则返回 None。
    """
    candidate_keys = ["ee_pose", "ee_poses", "end_effector_pose", "eef_pose", "target_pose"]
    points = []
    for result in path_list:
        if not isinstance(result, dict):
            continue
        for key in candidate_keys:
            if key not in result:
                continue
            raw = np.asarray(result[key], dtype=np.float64)
            if raw.ndim == 1 and raw.shape[0] >= 3:
                points.append(raw[:3])
            elif raw.ndim == 2 and raw.shape[1] >= 3:
                points.extend(raw[:, :3])
            break
    if len(points) < 2:
        return None
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _compute_episode_metrics(traj_data):
    left_pos = _flatten_joint_positions(traj_data.get("left_joint_path", []))
    right_pos = _flatten_joint_positions(traj_data.get("right_joint_path", []))

    joint_path_length = _calc_path_length(left_pos) + _calc_path_length(right_pos)
    smoothness = _calc_smoothness(left_pos) + _calc_smoothness(right_pos)

    left_ee_len = _calc_ee_path_length(traj_data.get("left_joint_path", []))
    right_ee_len = _calc_ee_path_length(traj_data.get("right_joint_path", []))
    if left_ee_len is None and right_ee_len is None:
        # 当前默认 planner 结果中通常仅含 joint trajectory，使用 joint path length 作为代理指标
        ee_path_length = joint_path_length
        ee_path_source = "joint_path_proxy"
    else:
        ee_path_length = (0.0 if left_ee_len is None else left_ee_len) + (0.0 if right_ee_len is None else right_ee_len)
        ee_path_source = "ee_pose"

    return {
        "joint_path_length": joint_path_length,
        "smoothness": smoothness,
        "ee_path_length": ee_path_length,
        "ee_path_source": ee_path_source,
        "step_num": int(left_pos.shape[0] + right_pos.shape[0]),
    }


def _load_traj_data(save_path, idx):
    file_path = os.path.join(save_path, "_traj_data", f"episode{idx}.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _select_high_quality_episodes(args, candidate_count):
    filter_cfg = args.get("episode_filter", {})
    target_num = args["episode_num"]
    weights = filter_cfg.get("weights", {})
    w_smooth = float(weights.get("smoothness", 1.0))
    w_joint_len = float(weights.get("joint_path_length", 0.3))
    w_ee_len = float(weights.get("ee_path_length", 0.7))

    metrics_list = []
    for idx in range(candidate_count):
        traj_data = _load_traj_data(args["save_path"], idx)
        metrics = _compute_episode_metrics(traj_data)
        cost = (w_smooth * metrics["smoothness"] + w_joint_len * metrics["joint_path_length"] + w_ee_len * metrics["ee_path_length"])
        metrics["quality_score"] = float(-cost)  # 分数越高越好
        metrics["source_episode_idx"] = idx
        metrics_list.append(metrics)

    metrics_list.sort(key=lambda x: x["quality_score"], reverse=True)
    selected = metrics_list[:target_num]
    selected_indices = [m["source_episode_idx"] for m in selected]

    filter_log = {
        "target_episode_num": target_num,
        "candidate_episode_num": candidate_count,
        "weights": {
            "smoothness": w_smooth,
            "joint_path_length": w_joint_len,
            "ee_path_length": w_ee_len,
        },
        "selected_indices": selected_indices,
        "selected_metrics": selected,
        "all_metrics": metrics_list,
    }
    return selected_indices, filter_log


def main(task_name=None, task_config=None):

    task = class_decorator(task_name)
    config_path = f"./task_config/{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []
    selected_indices = []
    filter_cfg = args.get("episode_filter", {})
    enable_filter = bool(filter_cfg.get("enable", False))
    target_episode_num = int(args["episode_num"])
    oversample_ratio = float(filter_cfg.get("oversample_ratio", 1.3))
    min_extra = int(filter_cfg.get("min_extra", 5))
    candidate_episode_num = target_episode_num
    if enable_filter:
        candidate_episode_num = max(target_episode_num + min_extra, int(math.ceil(target_episode_num * oversample_ratio)))

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        seed_cache_file = "seed_candidates.txt" if enable_filter else "seed.txt"
        seed_cache_path = os.path.join(args["save_path"], seed_cache_file)
        if os.path.exists(seed_cache_path):
            with open(seed_cache_path, "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < candidate_episode_num:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    print(f"simulate data episode {suc_num} success! (seed = {epid})")
                    seed_list.append(epid)
                    TASK_ENV.save_traj_data(suc_num)
                    suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(seed_cache_path, "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")

        if enable_filter:
            selected_indices, filter_log = _select_high_quality_episodes(args, candidate_count=suc_num)
            selected_seed_list = [seed_list[idx] for idx in selected_indices]
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in selected_seed_list:
                    file.write("%s " % sed)
            with open(os.path.join(args["save_path"], "episode_filter_info.json"), "w", encoding="utf-8") as file:
                json.dump(filter_log, file, ensure_ascii=False, indent=4)
            seed_list = selected_seed_list
            print(f"[Episode Filter] Select {len(selected_indices)}/{suc_num} high-quality episodes.")
        else:
            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)
            selected_indices = list(range(len(seed_list)))
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]
        if enable_filter and os.path.exists(os.path.join(args["save_path"], "episode_filter_info.json")):
            with open(os.path.join(args["save_path"], "episode_filter_info.json"), "r", encoding="utf-8") as file:
                filter_log = json.load(file)
            selected_indices = [int(i) for i in filter_log.get("selected_indices", list(range(len(seed_list))))]
        else:
            selected_indices = list(range(len(seed_list)))

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(args["save_path"], 'data', f'episode{idx}.hdf5')
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        final_episode_num = min(args["episode_num"], len(selected_indices), len(seed_list))
        for episode_idx in range(st_idx, final_episode_num):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            source_idx = selected_indices[episode_idx]
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(source_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video()
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"

        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(task_name=task_name, task_config=task_config)
