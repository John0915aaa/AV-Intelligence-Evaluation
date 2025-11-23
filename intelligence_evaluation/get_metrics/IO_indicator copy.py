"""
这个脚本用于计算IO
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

DELTA_T = 0.5
considertime = 5
starting_extension_time = 1
ending_extension_time = 2.5

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


def get_collision_point(column_dict, agents_states, AV_index, HV_index, time_index):
    position_index = [column_dict['x'], column_dict['y']]
    v_index = [column_dict['vx'], column_dict['vy']]

    timerange = considertime / 0.1

    #------------计算冲突点，以及两辆车到达冲突点的距离------------
    AV_processed_tracks = process_tracks_single(
        considertime, agents_states, AV_index, time_index,
        timerange, position_index, v_index)
    HV_processed_tracks = process_tracks_single(
        considertime, agents_states, HV_index, time_index,
        timerange, position_index, v_index)
    
    if AV_processed_tracks is None or HV_processed_tracks is None:
        return None, None, None, None, None, None
    
    AV_line, AV_speed = AV_processed_tracks['line'], AV_processed_tracks['velocity']
    HV_line, HV_speed = HV_processed_tracks['line'], HV_processed_tracks['velocity']

    intersection = AV_line.intersection(HV_line)
    print(f"intersection: {intersection}")

    if intersection.is_empty:
    #print(f"ego_position: {ego_x, ego_y}")
        return None, None, None, None, None, None
    #print(f"ego_position: {ego_x, ego_y}")
    #print(intersection)
    AV_agent_state = agents_states[AV_index, time_index, :]
    AV_agent_states = get_agent_state(column_dict, AV_agent_state)
    HV_agent_state = agents_states[HV_index, time_index, :]
    HV_agent_states = get_agent_state(column_dict, HV_agent_state)
    AV_x, AV_y = AV_agent_states.x, AV_agent_states.y
    HV_x, HV_y = HV_agent_states.x, HV_agent_states.y
    #print(f"s_position: {S_x, S_y}")
    #print(f"l_position: {x_l, y_l}")

    _, collision_point = nearest_points(Point(AV_x, AV_y), intersection)

    dis_AV, dis_HV = AV_line.project(collision_point), HV_line.project(collision_point)

    #------------计算未来Delta_t时刻内两辆车的行驶距离Delta_S------------    
    future_AV_agent_state = agents_states[AV_index, time_index+int(DELTA_T/0.1), :]
    future_AV_agent_states = get_agent_state(column_dict, future_AV_agent_state)
    AV_future_dis = AV_line.project(Point(future_AV_agent_states.x, future_AV_agent_states.y))

    return collision_point, dis_AV, dis_HV, AV_speed, HV_speed, AV_future_dis



def normalize_angle(angle):
    """
    将角度归一化到[-π, π]范围内
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def calculate_Snorm_for_AV(dis_AV, dis_HV, AV_speed, HV_speed, delta_t, AV_future_dis, delta_theta):
    

    if delta_theta > math.pi/2:
        #相对冲突
        delta_theta = math.pi - delta_theta
        #min:
        l_HV_min = dis_HV + length + width*(1/(math.sin(delta_theta)) + 1/(math.tan(delta_theta)))/2
        l_AV_min = dis_AV
        t_min = l_HV_min / HV_speed
        a_min = 2*(l_AV_min - AV_speed*t_min)/t_min**2

        #max:
        l_HV_max = dis_HV
        l_AV_max = dis_AV + length + width*(1/(math.sin(delta_theta)) + 1/(math.tan(delta_theta)))/2
        t_max = l_HV_max / HV_speed
        a_max = 2*(l_AV_max - AV_speed*t_max)/t_max**2
    else:
        #相向冲突
        #min:
        l_HV_min = dis_HV + length + width*(math.tan(delta_theta))/2
        l_AV_min = dis_AV
        t_min = l_HV_min / HV_speed
        a_min = 2*(l_AV_min - AV_speed*t_min)/t_min**2

        #max:
        l_HV_max = dis_HV
        l_AV_max = dis_AV + length + width*(math.tan(delta_theta))/2
        t_max = l_HV_max / HV_speed
        a_max = 2*(l_AV_max - AV_speed*t_max)/t_max**2
    
    S_min = -((AV_speed + a_min*delta_t)**2 - AV_speed**2)/(2*a_min)
    S_max = ((AV_speed + a_max*delta_t)**2 - AV_speed**2)/(2*a_max)

    if AV_future_dis >= S_max:
        S_norm = 1
    elif AV_future_dis <= S_min:
        S_norm = 0
    else:
        S_norm = (AV_future_dis - S_min) / (S_max - S_min)
    
    return S_norm

def calculate_ITSI(dis_AV, dis_HV, AV_speed, HV_speed):
    TTCP_AV = (dis_AV+length)/AV_speed
    TTCP_HV = (dis_HV+length)/HV_speed
    delta_TTCP = TTCP_AV - TTCP_HV

    if dis_HV >= HV_speed*TTCP_AV/2:
        a_c = 2*(dis_HV - HV_speed*TTCP_AV)/(TTCP_AV**2)
    else:
        a_c = HV_speed**2 / (2*dis_HV)
    
    delta_TTCP_norm = 1 - (1/(1+math.exp(-delta_TTCP)))
    a_c_norm = 1 - (1/(1+math.exp(-a_c)))

    softmax1 = math.exp(delta_TTCP_norm) / (math.exp(delta_TTCP_norm) + math.exp(a_c_norm))
    softmax2 = math.exp(a_c_norm) / (math.exp(delta_TTCP_norm) + math.exp(a_c_norm))

    ITSI = softmax1*delta_TTCP_norm + softmax2*a_c_norm
    return ITSI

IO_results = []

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

        '''if index > 300:
            continue'''


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
        print(dt)
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


        AV_index = all_agents.index(ego_id)
        HV_index = all_agents.index(key_agent)

        S_norm_s = []
        ITSI_s = []
        IO_s = []
        IO_s_data = {
                    'dataset': desired_data,
                    'folder': folder,
                    'scenario_idx': raw_scene_id,
                    'track_id': track_id,
                    'ego_id': ego_id,
                }
        for time_index, timestamp in enumerate(all_timesteps):
            
            print(f"timestamp = {timestamp}")
            print(f"time_index = {time_index}")

            collision_point, dis_AV, dis_HV, AV_speed, HV_speed, AV_future_dis = get_collision_point(column_dict, agents_states, AV_index, HV_index, time_index)

            if dis_AV is None:
                print(f"当前时刻无冲突点")
                continue
            
            AV_state = agents_states[AV_index, time_index, :]
            AV_states = get_agent_state(column_dict, AV_state)

            HV_state = agents_states[HV_index, time_index, :]
            HV_states = get_agent_state(column_dict, HV_state)

            AV_h, HV_h = AV_states.h, HV_states.h

            delta_theta = abs(normalize_angle(AV_h - HV_h))
            print(delta_theta * 180 / math.pi)

            S_norm = calculate_Snorm_for_AV(dis_AV, dis_HV, AV_speed, HV_speed, DELTA_T, AV_future_dis, delta_theta)
            ITSI = calculate_ITSI(dis_AV, dis_HV, AV_speed, HV_speed)

            IO = 1 - ITSI*S_norm

            IO_s_data[f"IO, t={timestamp}"] = IO

            print(f"S_norm = {S_norm}")
            print(f"ITSI = {ITSI}")
            print(f"IO = {IO}")

            """ S_norm_s.append(S_norm)
            ITSI_s.append(ITSI)
            IO_s.append(IO) """
            
        IO_results.append(IO_s_data)

        # ---------- Visualization -----------
        """ plt.figure(figsize=(10, 6))
        plt.plot(all_timesteps[:len(S_norm_s)], S_norm_s, label='S_norm', marker='o', linestyle='-')
        plt.plot(all_timesteps[:len(ITSI_s)], ITSI_s, label='ITSI', marker='s', linestyle='--')
        plt.plot(all_timesteps[:len(IO_s)], IO_s, label='IO', marker='d', linestyle='-.')
        
        plt.xlabel("Time Step")
        plt.ylabel("Indicator Value")
        plt.title(f"Interaction Indicators Over Time (Scene {scene_info.raw_scene_id})")
        plt.legend()
        plt.grid(True)
        plt.show(block=True) """

    IO_savepath = f"{save_dir}/IO_2.csv"
    save_results(IO_results, IO_savepath)

if __name__ == "__main__":
    target_id = None
    calculate_indicator(target_id)