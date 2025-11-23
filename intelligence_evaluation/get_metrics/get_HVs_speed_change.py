"""
计算背景车辆（HV）在 start_time 前后平均速度变化比例
修改于 2025-10-14
"""

import os
import pandas as pd
from trajdata import UnifiedDataset
from utils.trajdata_utils import DataFrameCache, get_agent_states
from utils.visualize_utils import get_map_and_kdtrees
import math
import numpy as np

considertime = 5
starting_extension_time = 1
ending_extension_time = 2.5

length = 4.5
width = 1.8

save_dir = os.path.join(os.getcwd(), "metrics_results/efficiency")
os.makedirs(save_dir, exist_ok=True)

interaction_idx_info = 'waymo_idx_all.csv'

FOLDER_CACHE_MAP = {
    'waymo_0-299': '/home/zjr/文档/InterHub_cache/waymo_0-299',
    'waymo_300-499': '/home/zjr/文档/InterHub_cache/waymo_300-499',
    'waymo_500-799': '/home/zjr/文档/InterHub_cache/waymo_500-799',
    'waymo_800-999': '/home/zjr/文档/InterHub_cache/waymo_800-999'
}

extract_df = pd.read_csv(interaction_idx_info)


class AgentState:
    def __init__(self, x, y, z, vx, vy, ax, ay, h):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay
        self.h = h


def get_agent_state(column_dict, agent_state):
    x = agent_state[column_dict['x']]
    y = agent_state[column_dict['y']]
    z = agent_state[column_dict['z']]
    vx = agent_state[column_dict['vx']]
    vy = agent_state[column_dict['vy']]
    ax = agent_state[column_dict['ax']]
    ay = agent_state[column_dict['ay']]
    h = agent_state[column_dict['heading']]
    return AgentState(x, y, z, vx, vy, ax, ay, h)


def get_dataset(desired_data, cache_location):
    dataset = UnifiedDataset(
        desired_data=[desired_data],
        standardize_data=False,
        rebuild_cache=False,
        rebuild_maps=False,
        centric="scene",
        verbose=True,
        cache_location=cache_location,
        num_workers=os.cpu_count(),
        incl_vector_map=True,
        data_dirs={desired_data: ' '}
    )
    return dataset


def calculate_indicator(target_id):
    results = []

    for rank, (idx, row) in enumerate(extract_df.iterrows(), start=1):
        desired_data = row['dataset']
        folder = row['folder']
        raw_scene_id = int(row['scenario_idx'])
        start = int(row['start'])
        end = int(row['end'])
        start_time = float(row.get('start_time', 0))
        track_id = row['track_id']
        interact_ids = track_id.split(';')
        index = row['index']
        key_agents = row['key_agents'].split(';')
        AV_included = row['AV_included']

        print(f"\n>>> Processing index {index}, scene {raw_scene_id}")

        cache_location = FOLDER_CACHE_MAP.get(folder)
        if cache_location is None:
            print(f"Unknown folder: {folder}, skipping.")
            continue

        """ if index >= 50:
            continue """

        if AV_included == "AV":
            ego_id = next((agent for agent in interact_ids if 'ego' in agent), None)
        else:
            i=1
            for agent in key_agents:
                if i==1:
                    i+=1
                else:
                    ego_id = agent

        dataset = get_dataset(desired_data, cache_location)

        # -------- 找到场景 --------
        id_rawid = {desired_scene.raw_data_idx: idx for idx, desired_scene in enumerate(dataset.scenes())}
        desired_scene = dataset.get_scene(id_rawid[raw_scene_id])

        dt = desired_scene.dt
        agents = {agent.name: agent for agent in desired_scene.agents}
        all_agents = list(agents.keys())

        first, last = 99999, 0
        for agent in interact_ids:
            first = min(first, agents[agent].first_timestep)
            last = max(last, agents[agent].last_timestep)
        interaction_start = max(first, int(start - starting_extension_time / dt))
        interaction_end = min(last, int(end + ending_extension_time / dt))
        all_timesteps = range(interaction_start, interaction_end)

        # 获取地图和缓存
        vec_map, lane_kd_tree = get_map_and_kdtrees(dataset, desired_scene)
        scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
        column_dict = scene_cache.column_dict

        # 获取状态数据
        agents_states, _ = get_agent_states(
            interact_ids, all_agents, vec_map, lane_kd_tree,
            scene_cache, desired_scene, column_dict, all_timesteps
        )

        # 获取 HV（非ego）
        vehicles = [agent for agent in desired_scene.agents if (agent.type == 1 and agent.name != ego_id)]

        row_result = {'index': index,
                      'start_time': start_time,
                      'type': AV_included,
                      'ego': ego_id,}

        # --- 计算每辆 HV 的速度变化比例 ---
        for i, vehicle in enumerate(vehicles, start=1):
            vehicle_index = all_agents.index(vehicle.name)
            timerange = range(interaction_start, interaction_end)

            # 计算 start_time 对应的帧索引
            split_frame = int((start_time - start * dt) / dt) if start_time > 0 else 0
            split_frame = max(0, min(split_frame, len(timerange) - 1))

            if start_time == 0:
                ratio = 0.0
            else:
                before_range = timerange[:split_frame]
                after_range = timerange[split_frame:]

                def calc_mean_speed(t_range):
                    if len(t_range) == 0:
                        return np.nan
                    v_sum, n = 0, 0
                    for time_idx, _ in enumerate(t_range):
                        car_state = agents_states[vehicle_index, time_idx, :]
                        car = get_agent_state(column_dict, car_state)
                        v = math.sqrt(car.vx ** 2 + car.vy ** 2)
                        if v <= 0.5:
                            continue
                        v_sum += v
                        n += 1
                    return v_sum / n if n > 0 else np.nan

                mean_before = calc_mean_speed(before_range)
                mean_after = calc_mean_speed(after_range)

                if np.isnan(mean_before) or mean_before == 0:
                    ratio = np.nan
                else:
                    ratio = max(0, (mean_before - mean_after) / mean_before)

            row_result[f'HV{i}'] = ratio

        results.append(row_result)

    # --- 保存结果 ---
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(save_dir, "HVs_speed_change_ratio.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved results to {out_csv}")


if __name__ == "__main__":
    target_id = None
    calculate_indicator(target_id)
