"""
这个脚本用于计算背景车辆的最大、最小平均速度
修改于2025-09-28
"""

import os
import pandas as pd
from trajdata import UnifiedDataset
from utils.trajdata_utils import DataFrameCache, get_agent_states
from utils.visualize_utils import get_map_and_kdtrees
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from utils.visualize_utils import process_tracks_single
import matplotlib.pyplot as plt
import math
import numpy as np

considertime = 5
starting_extension_time = 1
ending_extension_time = 2.5

length = 4.5
width = 1.8

save_dir = os.path.join(os.getcwd(), "results")
os.makedirs(save_dir, exist_ok=True)   # 自动创建，不存在就新建

interaction_idx_info = '/home/zjr/文档/智能度评价相关/数据向自然语言转换/指标/舒适/waymo_AV_selected.csv'

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
    
    def __repr__(self):
        return f"AgentState({self.x}, {self.y}, {self.z}, {self.vx}, {self.vy}, {self.ax}, {self.ay}, {self.h})"
    

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
        rebuild_cache=False,  # Do not rebuild cache
        rebuild_maps=False,   # Do not rebuild maps
        centric="scene",
        verbose=True,
        cache_location=cache_location,  # Ensure cache_location is a string
        num_workers=os.cpu_count(),
        incl_vector_map=True,
        data_dirs={desired_data: ' '}
    )

    return dataset


def save_results(results, save_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path)
    print(f"Results saved to {save_path}")


def calculate_meanspeed_in_currenttime(column_dict, agents_states, ego_index, time_index):
    ego_state = agents_states[ego_index, time_index, :]
    ego_states = get_agent_state(column_dict, ego_state)

    meanspeed_value = math.sqrt(ego_states.vx**2 + ego_states.vy**2)
    
    return meanspeed_value


def calculate_ego_states_for_comfortable(column_dict, agents_states, ego_index, time_index):
    ego_state = agents_states[ego_index, time_index, :]
    ego_states = get_agent_state(column_dict, ego_state)

    a_p = ego_states.ax*math.cos(ego_states.h) + ego_states.ay*math.sin(ego_states.h)
    a_l = -ego_states.ax*math.sin(ego_states.h) + ego_states.ay*math.cos(ego_states.h)

    v_p = ego_states.vx*math.cos(ego_states.h) + ego_states.vy*math.sin(ego_states.h)
    v_l = -ego_states.vx*math.sin(ego_states.h) + ego_states.vy*math.cos(ego_states.h)

    if v_p >= 0.01:
        yaw_rate = a_l / v_p
    else:
        yaw_rate = 0
    
    return a_p, a_l, v_p, v_l, yaw_rate



min_mean_speeds = []
max_mean_speeds = []
datas = []
def calculate_indicator(target_id):
    for rank, (idx, row) in enumerate(extract_df.iterrows(), start=1):
        desired_data = row['dataset']
        folder = row['folder']
        raw_scene_id = int(row['scenario_idx'])
        start = int(row['start'])
        end = int(row['end'])
        track_id = row['track_id']
        key_agents = row['key_agents'].split(';')
        interact_ids = track_id.split(';')
        index = row['index']

        """ if index > 10:
            continue """

        ego_id = next((agent for agent in interact_ids if 'ego' in agent), None)

        cache_location = FOLDER_CACHE_MAP.get(folder)
        if cache_location is None:
            print(f"Unknown folder: {folder}, skipping.")
            continue

        dataset = get_dataset(desired_data, cache_location)

        #--------找到符合要求的场景--------
        id_rawid = {desired_scene.raw_data_idx: idx for idx, desired_scene in enumerate(dataset.scenes())}
        desired_scene = dataset.get_scene(id_rawid[raw_scene_id])

        #-------计算交互的起始与结束时间-------
        dt = desired_scene.dt
        agents = {agent.name: agent for agent in desired_scene.agents}
        all_agents = list(agents.keys())
        #print(f"all agents: {all_agents}")
        first, last = 99999, 0
        for agent in interact_ids:
            first = min(first, agents[agent].first_timestep)
            last = max(last, agents[agent].last_timestep)
        interaction_start = max(first, int(start - starting_extension_time / dt))
        interaction_end = min(last, int(end + ending_extension_time / dt))
        all_timesteps = range(interaction_start, interaction_end)

        #-------获取场景地图数据-------
        vec_map, lane_kd_tree = get_map_and_kdtrees(dataset, desired_scene)
        scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
        column_dict = scene_cache.column_dict

        #-------获取场景内所有车辆的状态信息(所有states包含: x y z h vx vy ax ay)-------
        agents_states, _ = get_agent_states(
            interact_ids, all_agents, vec_map, lane_kd_tree, scene_cache, desired_scene,
            column_dict, all_timesteps
        )

        #print(f"all_agents: {all_agents}")
        #print(f"desired_scene.agents: {desired_scene.agents}")

        print(f"index: {index}, scenario: {raw_scene_id}")

        vehicles = [agent for agent in desired_scene.agents if (agent.type == 1 and agent.name != 'ego')]
        #print(vehicles)

        first0, last0 = 99999, 0
        mean_speeds = []
        for vehicle in vehicles:
            
            vehicle_name = vehicle.name

            vehicle_index = all_agents.index(vehicle_name)
            
            first1 = min(first0, vehicle.first_timestep)
            last1 = max(last0, vehicle.last_timestep)

            start_time = max(first1, int(start - starting_extension_time / dt))
            end_time = min(last1, int(end + ending_extension_time / dt))

            timerange = range(interaction_start, interaction_end)
            #print(f"timerange = {timerange}")
            #print(f"当前agent name: {vehicle_name}")
            #print(f"当前agent index: {vehicle_index}")

            if timerange == 0:
                #print(f"当前车辆timerange = 0, skip")
                continue

            sum_speed = 0
            n = 0

            v_flag = False
            vs = []
            for time_index, timestamp in enumerate(timerange):
                car_state = agents_states[vehicle_index, time_index, :]
                car_states = get_agent_state(column_dict, car_state)

                vx, vy = car_states.vx, car_states.vy

                v = math.sqrt(vx**2 + vy**2)

                if v <= 0.5:
                    #print(v)
                    v_flag = True
                    break

                sum_speed += v
                n += 1

                #vs.append(v)
            

            if v_flag == True:
                continue

            #print(vs)
            
            mean_speed = sum_speed / n
            #print(f"当前是{vehicle_name}, mean_speed = {mean_speed}")
            mean_speeds.append(mean_speed)

        #print(f"mean_speeds: {mean_speeds}")

        min_mean_speed = min(mean_speeds)
        max_mean_speed = max(mean_speeds)

        print(f"min_mean_speed = {min_mean_speed}, max_mean_speed = {max_mean_speed}\n\n")

        data = {
            'index': index,
            'min_mean_speed': min_mean_speed,
            'max_mean_speed': max_mean_speed,
        }

        datas.append(data)

    df = pd.DataFrame(datas)
    
    out_csv = os.path.join(save_dir, "HVs_mean_speed.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")



if __name__ == "__main__":
    target_id = None
    calculate_indicator(target_id)