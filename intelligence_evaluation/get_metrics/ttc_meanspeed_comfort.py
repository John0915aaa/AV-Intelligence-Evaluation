"""
这个脚本用于计算waymo数据集中AV车辆在交互过程中的表现情况, 包括量化TTC, 平均车速, 以及舒适性指标(纵向加速度, 横向加速度, 横向速度, 横摆角速度)
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
starting_extension_time = 1 #开始前延长时间
ending_extension_time = 2.5 #结束后延长时间

length = 4.5
width = 1.8

save_dir = os.path.join(os.getcwd(), "metrics_results")
os.makedirs(save_dir, exist_ok=True)   # 自动创建，不存在就新建

interaction_idx_info = 'waymo_idx_all.csv' #场景索引csv

FOLDER_CACHE_MAP = {
    'waymo_0-299': '/home/zjr/文档/InterHub_cache/waymo_0-299',
    'waymo_300-499': '/home/zjr/文档/InterHub_cache/waymo_300-499',
    'waymo_500-799': '/home/zjr/文档/InterHub_cache/waymo_500-799',
    'waymo_800-999': '/home/zjr/文档/InterHub_cache/waymo_800-999'
} #数据集路径
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
    results_df.to_csv(save_path, index=False)
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


from TwoDimTTC import TTC
def calculate_ttc_with_one_agent_in_currenttime(column_dict, agents_states, ego_index, agent_index, time_index):
    ego_state = agents_states[ego_index, time_index, :]
    ego_states = get_agent_state(column_dict, ego_state)

    agent_state = agents_states[agent_index, time_index, :]
    agent_states = get_agent_state(column_dict, agent_state)

    data = {'x_i': ego_states.x,
            'y_i': ego_states.y,
            'vx_i': ego_states.vx,
            'vy_i': ego_states.vy,
            'hx_i': math.cos(ego_states.h),
            'hy_i': math.sin(ego_states.h),
            'length_i': length,
            'width_i': width,
            'x_j': agent_states.x,
            'y_j': agent_states.y,
            'vx_j': agent_states.vx,
            'vy_j': agent_states.vy,
            'hx_j': math.cos(agent_states.h),
            'hy_j': math.sin(agent_states.h),
            'length_j': length,
            'width_j': width
    }

    samples = pd.DataFrame(data, index=[0])
    ttc_value = TTC(samples, 'values')

    return ttc_value


ttc_results = []
meanspeed_results = []
a_p_results = []
a_l_results = []
v_p_results = []
v_l_results = []
yaw_rate_results = []

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
        AV_included = row['AV_included']

        """ if index < 30192:
            continue """

        if AV_included == "AV":
            ego_id = next((agent for agent in interact_ids if 'ego' in agent), None)
            for agent in key_agents:
                if agent != 'ego':
                    key_agent = agent   
        else:
            i=1
            for agent in key_agents:
                if i==1:
                    key_agent = agent
                    i+=1
                else:
                    ego_id = agent

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

        ego_index = all_agents.index(ego_id)
        agent_index = all_agents.index(key_agent)

        #---------------calculate the ttc between ego and other agents---------------
        ttcs = []
        for time_index, timestamp in enumerate(all_timesteps):

            ttc_value = calculate_ttc_with_one_agent_in_currenttime(column_dict, agents_states, ego_index, agent_index, time_index)
            ttcs.append(ttc_value)
        
        ttc_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': track_id,
            'ego_id': ego_id,
            'timerange': all_timesteps,
        }
        for time_index, timestamp in enumerate(all_timesteps):
            ttc_data[f"v, t={timestamp}"] = ttcs[time_index]

        ttc_results.append(ttc_data)

        #---------------calculate mean speed of ego---------------
        speeds = []
        for time_index, timestamp in enumerate(all_timesteps):
            speed_value = calculate_meanspeed_in_currenttime(column_dict, agents_states, ego_index, time_index)
            speeds.append(speed_value)
        mean_speed = np.sum(speeds)/len(speeds)

        meanspeed_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': track_id,
            'ego_id': ego_id,
            'timerange': all_timesteps,
            'mean_speed': mean_speed,
        }
        for time_index, timestamp in enumerate(all_timesteps):
            meanspeed_data[f"v, t={timestamp}"] = speeds[time_index]

        meanspeed_results.append(meanspeed_data)


        #---------------calculate comfortable states of ego---------------
        a_p_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': interact_ids,
            'ego_id': ego_id,
            'timerange': all_timesteps,
        }
        a_l_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': interact_ids,
            'ego_id': ego_id,
            'timerange': all_timesteps,
        }
        v_p_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': interact_ids,
            'ego_id': ego_id,
            'timerange': all_timesteps,
        }
        v_l_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': interact_ids,
            'ego_id': ego_id,
            'timerange': all_timesteps,
        }
        yaw_rate_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'track_id': interact_ids,
            'ego_id': ego_id,
            'timerange': all_timesteps,
        }
        for time_index, timestamp in enumerate(all_timesteps):
            a_p, a_l, v_p, v_l, yaw_rate = calculate_ego_states_for_comfortable(column_dict, agents_states, ego_index, time_index)
            a_p_data[f'ap, t={timestamp}'] = a_p
            a_l_data[f'al, t={timestamp}'] = a_l
            v_p_data[f'vp, t={timestamp}'] = v_p
            v_l_data[f'vl, t={timestamp}'] = v_l
            yaw_rate_data[f'yr, t={timestamp}'] = yaw_rate

        a_p_results.append(a_p_data)#纵向加速度
        a_l_results.append(a_l_data)#横向加速度
        v_p_results.append(v_p_data)#纵向速度
        v_l_results.append(v_l_data)#横向速度
        yaw_rate_results.append(yaw_rate_data)#横摆角速度

    ttc_savepath = f"{save_dir}/ttc_results.csv"
    save_results(ttc_results, ttc_savepath)

    meanspeed_savepath = f"{save_dir}/meanspeed_results.csv"
    save_results(meanspeed_results, meanspeed_savepath)

    ap_savepath = f"{save_dir}/ap_results.csv"
    save_results(a_p_results, ap_savepath)
    al_savepath = f"{save_dir}/al_results.csv"
    save_results(a_l_results, al_savepath)
    vp_savepath = f"{save_dir}/vp_results.csv"
    save_results(v_p_results, vp_savepath)
    vl_savepath = f"{save_dir}/vl_results.csv"
    save_results(v_l_results, vl_savepath)
    yawrate_savepath = f"{save_dir}/yawrate_results.csv"
    save_results(yaw_rate_results, yawrate_savepath)


if __name__ == "__main__":
    target_id = None
    calculate_indicator(target_id)