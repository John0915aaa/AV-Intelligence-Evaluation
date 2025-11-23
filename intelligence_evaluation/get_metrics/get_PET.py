"""

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
from pathlib import Path


DELTA_T = 2
# 检查文件是否已存在（如果不存在，创建 CSV 并写入列名）

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


def get_collision_point_with_dis_and_speed(column_dict, agents_states, S_agent_index, L_agent_index, time_index):
    position_index = [column_dict['x'], column_dict['y']]
    v_index = [column_dict['vx'], column_dict['vy']]

    timerange = considertime / 0.1

    #------------计算冲突点，以及两辆车到达冲突点的距离------------
    S_processed_tracks = process_tracks_single(
        considertime, agents_states, S_agent_index, time_index,
        timerange, position_index, v_index)
    L_processed_tracks = process_tracks_single(
        considertime, agents_states, L_agent_index, time_index,
        timerange, position_index, v_index)
    
    if S_processed_tracks is None or L_processed_tracks is None:
        return None, None, None, None, None, None, None
    
    S_line, S_speed = S_processed_tracks['line'], S_processed_tracks['velocity']
    L_line, L_speed = L_processed_tracks['line'], L_processed_tracks['velocity']

    intersection = S_line.intersection(L_line)
    #print(f"intersection: {intersection}")

    if intersection.is_empty:
    #print(f"ego_position: {ego_x, ego_y}")
        return None, None, None, None, None, None, None
    #print(f"ego_position: {ego_x, ego_y}")
    #print(intersection)
    S_agent_state = agents_states[S_agent_index, time_index, :]
    S_agent_states = get_agent_state(column_dict, S_agent_state)
    L_agent_state = agents_states[L_agent_index, time_index, :]
    L_agent_states = get_agent_state(column_dict, L_agent_state)
    S_x, S_y = S_agent_states.x, S_agent_states.y
    L_x, L_y = L_agent_states.x, L_agent_states.y
    #print(f"s_position: {S_x, S_y}")
    #print(f"l_position: {x_l, y_l}")

    _, collision_point = nearest_points(Point(S_x, S_y), intersection)

    dis_S, dis_L = S_line.project(collision_point), L_line.project(collision_point)

    #------------计算未来Delta_t时刻内两辆车的行驶距离Delta_S------------    
    future_S_agent_state = agents_states[S_agent_index, time_index+DELTA_T, :]
    future_S_agent_states = get_agent_state(column_dict, future_S_agent_state)
    future_L_agent_state = agents_states[L_agent_index, time_index+DELTA_T, :]
    future_L_agent_states = get_agent_state(column_dict, future_L_agent_state)

    delta_S = S_line.project(Point(future_S_agent_states.x, future_S_agent_states.y))
    delta_L = L_line.project(Point(future_L_agent_states.x, future_L_agent_states.y))

    return collision_point, dis_S, dis_L, S_speed, L_speed, delta_S, delta_L


datas = []
PET_results = []
def calculate_indicator(target_id, traj_folder):
    #top_n = 5 # 选择交互强度最大的前x个场景
    #top_rows = extract_df.nlargest(top_n, 'PET')
    #for rank, (idx, row) in enumerate(top_rows.iterrows(), start=1):   #按指定的顺序进行迭代(如intensity, PET...)
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
        path_relationship = row['path_relationship']


        if index != 1 and index!= 22607 and index != 5035 and index!=10268:
            continue

        turn_label = row['turn_label'].split('-')
                    
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

        if row['path_relationship'] != 'CP':
            pet_data = {
            'index': index,
            'dataset': desired_data,
            'folder': folder,
            'scenario_idx': raw_scene_id,
            'type': path_relationship,
            'track_id': track_id,
            'ego_id': ego_id,
            'timerange': all_timesteps,
            'min_PET': np.inf,
        }
            PET_results.append(pet_data)
        else:

    #------------------------------------------------------------init_prompt------------------------------------------------------------
            ego_index = all_agents.index(ego_id)
            agent_index = all_agents.index(key_agent)

            ego_init_state = agents_states[ego_index, 0, :]
            AV_init_x, AV_init_y, AV_init_h = ego_init_state[column_dict['x']], ego_init_state[column_dict['y']], ego_init_state[column_dict['heading']]

            agent_init_state = agents_states[agent_index, 0, :]
            HV_init_x, HV_init_y, HV_init_h = agent_init_state[column_dict['x']], agent_init_state[column_dict['y']], agent_init_state[column_dict['heading']]

    #------------------------------------------------------------寻找冲突点------------------------------------------------------------
            for time_index, timestamp in enumerate(all_timesteps):

                collision_point, dis_AV, dis_HV, AV_speed, HV_speed, delta_AV, delta_HV = get_collision_point_with_dis_and_speed(column_dict, agents_states, ego_index, agent_index, time_index)

                if collision_point is not None:
                    break

    #------------------------------------------------------------以相对于冲突点的距离定义交互开始时间------------------------------------------------------------
            start_time = -1
            end_time = 9999
            start_time_index = 0
            for time_index, timestamp in enumerate(all_timesteps):
                ego_state = agents_states[ego_index, time_index, :]
                AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
                AV_dis = collision_point.distance(Point(AV_x, AV_y))

                agent_state = agents_states[agent_index, time_index, :]
                HV_x, HV_y = agent_state[column_dict['x']], agent_state[column_dict['y']]
                HV_dis = collision_point.distance(Point(HV_x, HV_y))

                dis_threshold = 15 #更改阈值以更改判定交互开始的方式
                if AV_dis is not None and HV_dis is not None:
                    if AV_dis <= dis_threshold or HV_dis <= dis_threshold:
                        start_time = timestamp
                        start_time_index = time_index
                        break

            PETs = []
            AV_pre_x, AV_pre_y = AV_init_x, AV_init_y
            HV_pre_x, HV_pre_y = HV_init_x, HV_init_y

            for time_index, timestamp in enumerate(all_timesteps):
                if timestamp <= start_time:
                    continue
                ego_state = agents_states[ego_index, time_index, :]
                AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
                AV_h = ego_state[column_dict['heading']]
                AV_v = ego_state[column_dict['vx']] * math.cos(AV_h) + ego_state[column_dict['vy']] * math.sin(AV_h)

                agent_state = agents_states[agent_index, time_index, :]
                HV_x, HV_y = agent_state[column_dict['x']], agent_state[column_dict['y']]
                HV_h = agent_state[column_dict['heading']]
                HV_v = agent_state[column_dict['vx']] * math.cos(HV_h) + agent_state[column_dict['vy']] * math.sin(HV_h)
                HV_dis = collision_point.distance(Point(HV_x, HV_y))

                t_AV_pass = (AV_dis + width/2)/ AV_v
                t_AV_arrive = (AV_dis - length/2 - width/2)/ AV_v
                t_HV_pass = (HV_dis + width/2) / HV_v
                t_HV_arrive = (HV_dis - length/2 - width/2) / HV_v

                if t_AV_pass > t_HV_arrive:
                    PET = t_AV_pass - t_HV_arrive
                elif t_HV_pass > t_AV_arrive:
                    PET = t_HV_pass - t_AV_arrive
                
                PETs.append(round(PET,3))

                if time_index > 0:
                    AV_v1 = np.array([collision_point.x-AV_pre_x, collision_point.y-AV_pre_y])
                    AV_v2 = np.array([collision_point.x-AV_x, collision_point.y-AV_y])
                    if np.dot(AV_v1, AV_v2) < 0:
                        end_time = timestamp
                        break

                    HV_v1 = np.array([collision_point.x-HV_pre_x, collision_point.y-HV_pre_y])
                    HV_v2 = np.array([collision_point.x-HV_x, collision_point.y-HV_y])
                    if np.dot(HV_v1, HV_v2) < 0:
                        end_time = timestamp
                        break
            
            min_PET = min(PETs)
            lens = len(PETs)
            print(f"length = {lens}")
            print(start_time_index)
            pet_data = {
                'index': index,
                'dataset': desired_data,
                'folder': folder,
                'scenario_idx': raw_scene_id,
                'type': path_relationship,
                'track_id': track_id,
                'ego_id': ego_id,
                'timerange': all_timesteps,
                'min_PET': min_PET,
            }
            for time_index, timestamp in enumerate(all_timesteps):

                if timestamp >= start_time and timestamp < end_time:
                    
                    print(time_index)
                    pet_data[f"v, t={timestamp}"] = PETs[time_index - start_time_index]

            PET_results.append(pet_data)
        
    df = pd.DataFrame(PET_results)


    # 写入 CSV 文件（不写索引列）
    out_csv = os.path.join(save_dir, "CP_PET.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

if __name__ == "__main__":
    target_id = None
    traj_folder = "base_data/waymo_train"
    calculate_indicator(target_id, traj_folder)
