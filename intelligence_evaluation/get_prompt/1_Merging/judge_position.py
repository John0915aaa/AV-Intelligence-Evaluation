"""
该脚本专用于Merging场景的场景类型判断
主要判断车辆的切入类型, 分为: AV从HV后方切入, AV从HV前方切入, HV从AV后方切入, HV从AV前方切入
然后将交互类型结果存储到场景索引csv中
其中, 仅"AV从HV后方切入"和"HV从AV前方切入"类型被选择, 因为剩余两种类型并未考察到AV的智能度
修改于2025-09-28
"""

import os
import math
import pandas as pd
import numpy as np
from trajdata import UnifiedDataset
from utils.trajdata_utils import DataFrameCache, get_agent_states
from utils.visualize_utils import get_map_and_kdtrees
from TwoDimTTC import TTC
from collections import Counter

starting_extension_time = 1
ending_extension_time = 2.5
interaction_idx_info = 'waymo_index/waymo_MP2.csv' #场景索引csv

length = 4.5
width = 1.8

FOLDER_CACHE_MAP = {
    'waymo_0-299': '/home/zjr/文档/InterHub_cache/waymo_0-299',
    'waymo_300-499': '/home/zjr/文档/InterHub_cache/waymo_300-499',
    'waymo_500-799': '/home/zjr/文档/InterHub_cache/waymo_500-799',
    'waymo_800-999': '/home/zjr/文档/InterHub_cache/waymo_800-999'
}

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

def get_dataset(desired_data, cache_location):
    return UnifiedDataset(
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

def get_agent_state(column_dict, agent_state):
    return AgentState(
        round(agent_state[column_dict['x']],3),
        round(agent_state[column_dict['y']],3),
        round(agent_state[column_dict['z']],3),
        round(agent_state[column_dict['vx']],4),
        round(agent_state[column_dict['vy']],4),
        round(agent_state[column_dict['ax']],4),
        round(agent_state[column_dict['ay']],4),
        round(agent_state[column_dict['heading']],6),
    )

def generate_points_along_line(points, p, num_points):
    """
    根据给定的点p，在对应的车道点序列中，找到p前后各num_points个点。
    
    参数:
        points: 一个二维数组，包含车道点的序列，每个点是[x, y, z]。
        p: 一个三维坐标点[p_x, p_y, p_z]，需要找到其在车道中的位置。
        num_points: 需要生成的点的数量。
        
    返回:
        all_points: 生成的点序列。
    """
    # 计算点p在points中的索引
    p_idx = np.argmin(np.linalg.norm(points[:, :3] - p[:3], axis=1))

    # 确保获取点时不越界，前后各num_points个点
    start_idx = max(p_idx - num_points, 0)
    end_idx = min(p_idx + num_points, len(points) - 1)

    # 提取点序列
    all_points = points[start_idx:end_idx+1]

    return all_points


def find_closest_point_and_generate(p, lane_edge_points, num_points=15):
    """
    给定点p1（左边缘点）和p2（右边缘点），找到最接近的点，并生成前后num_points个点。
    
    参数：
        p1: 左车道上的一个点。
        p2: 右车道上的一个点。
        lane_left_points: 左车道的点序列。
        lane_right_points: 右车道的点序列。
        num_points: 生成的点数量（前后各num_points个点）。

    返回：
        left_points: 生成的左车道点序列。
        right_points: 生成的右车道点序列。
    """
    # 找到距离p最近的左边缘点
    distances = np.linalg.norm(lane_edge_points[:, :3] - p[:3], axis=1)
    closest_index = np.argmin(distances)
    closest_point = lane_edge_points[closest_index, :3]
    
    # 生成左车道和右车道的前后num_points个点
    lane_edge_points = generate_points_along_line(lane_edge_points, closest_point, num_points)
    
    return lane_edge_points

def get_closest_points_for_lane_edge(p_3d, lane_edge_points):
    """
    计算ego某个角点与道路lane左/右边缘的最近的点以及最小距离，
    并返回当前点及其下一个点（如果当前点是最后一个点，则返回上一个点）。

    Args:
        p_3d: ego当前时刻某一个角点的3d坐标(np.array([x, y, z]))
        lane_edge: 道路边缘的PolyLine对象，包含多个点
        
    Returns:
        closest_point: 距离当前角点最近的一个道路边缘坐标(np.array([lane_edge_x, lane_edge_y]))
        next_point: 距离当前角点最近的下一个点，若当前点是最后一个点，则返回上一个点
    """

    # 获取当前角点距离路径上各点的最近点

    distances = np.linalg.norm(lane_edge_points[:, :3] - p_3d[:3], axis=1)
    idx = np.argmin(distances)
    closest_point = lane_edge_points[idx, :3]

    if idx != len(lane_edge_points) - 1:
        next_id = idx + 1
        prev_id = idx - 1
        next_point_1 = lane_edge_points[next_id]
        prev_point_1 = lane_edge_points[prev_id]
        d_next = np.sqrt((closest_point[0]-next_point_1[0])**2 + (closest_point[1]-next_point_1[1])**2 + (closest_point[2]-next_point_1[2])**2)
        d_prev = np.sqrt((closest_point[0]-prev_point_1[0])**2 + (closest_point[1]-prev_point_1[1])**2 + (closest_point[2]-prev_point_1[2])**2)
        if d_next <= d_prev:
            closest_point_3d = closest_point
            next_point_3d = next_point_1
        else:
            closest_point_3d = prev_point_1
            next_point_3d = closest_point

    # 如果当前点是路径中的最后一个点，返回当前点和前一个点
    if idx == len(lane_edge_points) - 1:
        closest_point_3d = lane_edge_points[idx - 1]
        next_point_3d = closest_point
    elif idx == 0:
        closest_point_3d = closest_point
        next_point_3d = lane_edge_points[idx + 1]
        

    # 返回最近的点和下一个点
    closest_point = np.array([closest_point_3d[0], closest_point_3d[1]])
    next_point = np.array([next_point_3d[0], next_point_3d[1]])

    return closest_point, next_point

def calculate_vehicle_bounding_box(x, y, length, width, heading):
    """
    计算车辆的外包矩形四个角的坐标。
    假设车辆的中心在 (x, y)，朝向为 heading (弧度)，车辆的尺寸为 length 和 width。
    
    返回：四个角的坐标 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    # 计算车辆的半长和半宽
    half_length = length / 2
    half_width = width / 2

    # 车辆的四个角点，按顺时针方向
    corners = [
        (-half_length, half_width),
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
    ]
    
    # 旋转矩阵计算
    rotated_corners = []
    for corner in corners:
        dx, dy = corner
        # 将每个角点的坐标旋转以匹配车辆的朝向
        rotated_x = x + dx * math.cos(heading) - dy * math.sin(heading)
        rotated_y = y + dx * math.sin(heading) + dy * math.cos(heading)
        rotated_corners.append((rotated_x, rotated_y))

    return rotated_corners

def current_lane_for_ego(vec_map, column_dict, ego_state, length=length, width=width):
    """
    根据当前ego中心点坐标及车辆四个角点坐标获取ego所在道路，并返回当前道路，前后相邻道路合并后的结果
    """
    ego_lane = None
    x = ego_state[column_dict['x']]
    y = ego_state[column_dict['y']]
    z = ego_state[column_dict['z']]
    h = ego_state[column_dict['heading']]
    
    # 计算车辆的四个角点
    corners = calculate_vehicle_bounding_box(x, y, length, width, h)
    
    # 获取中心点所在的车道
    lanes_center = vec_map.get_current_lane(np.array([x, y, z, h]))
    
    # 获取每个角点所在的车道
    lane_counts = []
    
    # 先统计中心点的车道
    if len(lanes_center) >= 1:
        for lane_center in lanes_center:
            #print(f"lane_center: {lane_center.id}")
            lane_counts.append(lane_center.id)

    # 遍历四个角点，统计所在车道
    for corner in corners:
        corner_x, corner_y = corner
        lanes_for_corner = vec_map.get_current_lane(np.array([corner_x, corner_y, z, h]))
        
        if len(lanes_for_corner) >= 1:
            for lane_for_corner in lanes_for_corner:
                lane_counts.append(lane_for_corner.id)
        elif len(lanes_for_corner) == 0:
            closest_lanes = vec_map.get_lanes_within(np.array([corner_x, corner_y, z]), 2)
            if len(closest_lanes) >= 1 :
                for closest_lane in closest_lanes:
                    lane_counts.append(closest_lane.id)

    if lane_counts != []:
        # 使用Counter统计每个车道的点数，选择最多的那个车道
        lane_counter = Counter(lane_counts)
        #print(lane_counter)
        #print(lane_counts)
        most_common_lane_id = lane_counter.most_common(1)[0][0]
        #print(most_common_lane_id)
        
        ego_lane = vec_map.get_road_lane(most_common_lane_id)
    
    return ego_lane

def normalize_angle(angle):
    """
    将角度归一化到[-π, π]范围内
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_angle(x1, y1, x2, y2, h):
    """计算两点连线的角度与车头方向的夹角"""
    h_AB = math.atan2(y2 - y1, x2 - x1)
    return normalize_angle(h_AB - h)

def calculate_ttc_with_one_agent_in_currenttime(column_dict, agents_states, ego_index, agent_index, time_index):
    ego = get_agent_state(column_dict, agents_states[ego_index, time_index, :])
    agent = get_agent_state(column_dict, agents_states[agent_index, time_index, :])

    data = {
        'x_i': ego.x, 'y_i': ego.y, 'vx_i': ego.vx, 'vy_i': ego.vy,
        'hx_i': math.cos(ego.h), 'hy_i': math.sin(ego.h),
        'length_i': length, 'width_i': width,
        'x_j': agent.x, 'y_j': agent.y, 'vx_j': agent.vx, 'vy_j': agent.vy,
        'hx_j': math.cos(agent.h), 'hy_j': math.sin(agent.h),
        'length_j': length, 'width_j': width
    }
    samples = pd.DataFrame(data, index=[0])
    return TTC(samples, 'values')


def calculate_indicator(target_id=None, top_n=4):
    extract_df = pd.read_csv(interaction_idx_info, usecols=['index', 'dataset', 'folder', 'scenario_idx', 'track_id', 'start', 'end', 'intensity', 'PET', 'two/multi', 'vehicle_type', 'AV_included', 'key_agents', 'path_relationship', 'pre_post_direction', 'turn_label', 'priority_label'])
    #extract_df['lane_change_type'] = -1  # 初始化新列
    #extract_df['lane_change_time'] = -1
    #top_rows = extract_df.nsmallest(top_n, 'PET')
    #top_n = 4 # 选择交互强度最大的前x个场景
    #for rank, (idx, row) in enumerate(top_rows.iterrows(), start=1):   #按指定的顺序进行迭代(如intensity, PET...)
    for rank, (idx, row) in enumerate(extract_df.iterrows(), start=1):
        desired_data = row['dataset']
        folder = row['folder']
        raw_scene_id = int(row['scenario_idx'])
        start = int(row['start'])
        end = int(row['end'])
        track_id = row['track_id']
        index = row['index']

        key_agents = row['key_agents'].split(';')
        for agent in key_agents:
            if agent != 'ego':
                key_agent = agent
        

        """ if target_id is not None and raw_scene_id != target_id:
            continue """

        cache_location = FOLDER_CACHE_MAP.get(folder)
        if cache_location is None:
            print(f"Unknown folder: {folder}, skipping.")
            continue

        print(f"\nProcessing scene {raw_scene_id}...")

        interact_ids = track_id.split(';')
        ego_id = next((agent for agent in interact_ids if 'ego' in agent), None)
        agent_ids = [a for a in interact_ids if a != ego_id]

        cache_location = FOLDER_CACHE_MAP.get(folder)
        if cache_location is None:
            print(f"Unknown folder: {folder}, skipping.")
            continue

        print(f"\nProcessing scene {raw_scene_id}...")

        interact_ids = track_id.split(';')
        #ego_id = next((agent for agent in interact_ids if 'ego' in agent), None)
        agent_ids = [a for a in interact_ids if a != ego_id]

        dataset = get_dataset(desired_data, cache_location)

        id_rawid = {s.raw_data_idx: idx for idx, s in enumerate(dataset.scenes())}
        if raw_scene_id not in id_rawid:
            print(f"Scene {raw_scene_id} not found in dataset.")
            continue

        desired_scene = dataset.get_scene(id_rawid[raw_scene_id])
        vec_map, lane_kd_tree = get_map_and_kdtrees(dataset, desired_scene)
        scene_cache = DataFrameCache(dataset.cache_path, desired_scene)
        column_dict = scene_cache.column_dict

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

        agents_states, _ = get_agent_states(
            interact_ids, all_agents, vec_map, lane_kd_tree, scene_cache, desired_scene,
            column_dict, all_timesteps
        )

        ego_index = all_agents.index(ego_id)
        agent_index = all_agents.index(key_agent)

        type = None
        start_time = -1
        start_time_index = -1
        flag = 'None'
        for time_index, timestamp in enumerate(all_timesteps):
            ego = get_agent_state(column_dict, agents_states[ego_index, time_index, :])
            x, y, z, h = ego.x, ego.y, ego.z, ego.h
            #左上 右上
            corners = calculate_vehicle_bounding_box(x, y, length, width, h)
            left_point_up, right_point_up = corners[3], corners[2]

            ego_state = agents_states[ego_index, time_index, :]
            ego_lane = current_lane_for_ego(vec_map, column_dict, ego_state, length=length, width=width)


            if ego_lane is not None:
                #ego_left
                ego_lane_left = ego_lane.left_edge
                if ego_lane_left is not None:
                    left_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), ego_lane_left.points)
                    closest_point, next_point = get_closest_points_for_lane_edge(np.array([left_point_up[0], left_point_up[1], z]), left_lane_points_num)

                    x1, y1 = closest_point[0], closest_point[1]
                    x2, y2 = next_point[0], next_point[1]
                    angle_diff = calculate_angle(x1, y1, x2, y2, h)
                    
                    if abs(angle_diff) > math.pi / 2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    
                    A, B = np.array([x1, y1]), np.array([x2, y2])
                    C = (x2 - x1) * (left_point_up[1] - y1) - (y2 - y1) * (left_point_up[0] - x1)
                    if C > 0:
                        #print(f"AV正向左变道, 当前时刻: {timestamp}")
                        type = 1
                        start_time = timestamp
                        start_time_index = time_index
                        break

                #ego_right
                ego_lane_right = ego_lane.right_edge
                if ego_lane_right is not None:
                    rigt_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), ego_lane_right.points)
                    closest_point, next_point = get_closest_points_for_lane_edge(np.array([right_point_up[0], right_point_up[1], z]), rigt_lane_points_num)
                    x1, y1 = closest_point[0], closest_point[1]
                    x2, y2 = next_point[0], next_point[1]
                    angle_diff = calculate_angle(x1, y1, x2, y2, h)
                    
                    if abs(angle_diff) > math.pi / 2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    
                    A, B = np.array([x1, y1]), np.array([x2, y2])
                    C = (x2 - x1) * (right_point_up[1] - y1) - (y2 - y1) * (right_point_up[0] - x1)
                    if C < 0:
                        #print(f"AV正向右变道, 当前时刻: {timestamp}")
                        type = 1
                        start_time = timestamp
                        start_time_index = time_index
                        break

            agent = get_agent_state(column_dict, agents_states[agent_index, time_index, :])
            x, y, z, h = agent.x, agent.y, agent.z, agent.h
            #左上 右上
            corners = calculate_vehicle_bounding_box(x, y, length, width, h)
            left_point_up, right_point_up = corners[3], corners[2]

            agent_state = agents_states[agent_index, time_index, :]
            agent_lane = current_lane_for_ego(vec_map, column_dict, agent_state, length=length, width=width)

            if agent_lane is not None:
                #ego_left
                agent_lane_left = agent_lane.left_edge
                if agent_lane_left is not None: 
                    left_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), agent_lane_left.points)
                    closest_point, next_point = get_closest_points_for_lane_edge(np.array([left_point_up[0], left_point_up[1], z]), left_lane_points_num)

                    x1, y1 = closest_point[0], closest_point[1]
                    x2, y2 = next_point[0], next_point[1]
                    angle_diff = calculate_angle(x1, y1, x2, y2, h)
                    
                    if abs(angle_diff) > math.pi / 2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    
                    A, B = np.array([x1, y1]), np.array([x2, y2])
                    C = (x2 - x1) * (left_point_up[1] - y1) - (y2 - y1) * (left_point_up[0] - x1)
                    if C >= 0:
                        #print(f"HV正向左变道, 当前时刻: {timestamp}")
                        type = 2
                        start_time = timestamp
                        start_time_index = time_index
                        break

                #ego_right
                agent_lane_right = agent_lane.right_edge
                if agent_lane_right is not None:
                    rigt_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), agent_lane_right.points)
                    closest_point, next_point = get_closest_points_for_lane_edge(np.array([right_point_up[0], right_point_up[1], z]), rigt_lane_points_num)
                    x1, y1 = closest_point[0], closest_point[1]
                    x2, y2 = next_point[0], next_point[1]
                    angle_diff = calculate_angle(x1, y1, x2, y2, h)
                    
                    if abs(angle_diff) > math.pi / 2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    
                    A, B = np.array([x1, y1]), np.array([x2, y2])
                    C = (x2 - x1) * (right_point_up[1] - y1) - (y2 - y1) * (right_point_up[0] - x1)
                    if C <= 0:
                        #print(f"HV正向右变道, 当前时刻: {timestamp}")
                        type = 2
                        start_time = timestamp
                        start_time_index = time_index
                        break

        if start_time_index >= 0:
            ego = get_agent_state(column_dict, agents_states[ego_index, start_time_index, :])
            agent = get_agent_state(column_dict, agents_states[agent_index, start_time_index, :])

            dx = agent.x - ego.x
            dy = agent.y - ego.y
            proj = dx * math.cos(ego.h) + dy * math.sin(ego.h)
            if proj > 1.0:
                if type == 1:
                    print(f"AV在HV后变道, 当前时刻: {start_time}")
                    flag = 'AV_back'
                elif type == 2:
                    print(f"HV在AV前变道, 当前时刻: {start_time}")
                    flag = 'HV_front'
            elif proj < -1.0:
                if type == 1:
                    print(f"AV在HV前变道, 当前时刻: {start_time}")
                    flag = 'AV_front'
                elif type == 2:
                    print(f"HV在AV后变道, 当前时刻: {start_time}")
                    flag = 'HV_back'
            else:
                print(f"并列")
                flag = 'None'
        else:
            print("不存在变道行为")
            flag = 'None'
        
        extract_df.at[idx, 'lane_change_type'] = flag
        extract_df.at[idx, 'lane_change_time'] = start_time
        extract_df.at[idx, 'lane_change_time_index'] = start_time_index
        extract_df.to_csv(interaction_idx_info, index=False)





        end_time = -1
        end_time_index = -1
        if type == 2:
            for time_index, timestamp in enumerate(all_timesteps):
                flag1 = '左上 线外'
                flag2 = '右上 线外'
                flag3 = '右下 线外'
                flag4 = '左下 线外'

                pass_flag1 = False
                pass_flag2 = False
                pass_flag3 = False
                pass_flag4 = False
                agent = get_agent_state(column_dict, agents_states[agent_index, time_index, :])
                x, y, z, h = agent.x, agent.y, agent.z, agent.h
                #左上 右上
                corners = calculate_vehicle_bounding_box(x, y, length, width, h)
                left_point_up, right_point_up, right_point_down, left_point_down = corners[3], corners[2], corners[1], corners[0]

                agent_state = agents_states[agent_index, time_index, :]
                agent_lane = current_lane_for_ego(vec_map, column_dict, agent_state, length=length, width=width)
                #agent_lane = vec_map.get_closest_lane(np.array([x, y, z]))
                
                if agent_lane is not None:
                    #left
                    agent_lane_left = agent_lane.left_edge
                    if agent_lane_left is not None: 
                        left_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), agent_lane_left.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([left_point_up[0], left_point_up[1], z]), left_lane_points_num)

                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (left_point_up[1] - y1) - (y2 - y1) * (left_point_up[0] - x1)
                        if C < 0:
                            flag1 = '左上 线内'
                            if time_index >= start_time_index:
                                pass_flag1 = True
                    
                    if agent_lane_left is not None: 
                        left_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), agent_lane_left.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([left_point_down[0], left_point_down[1], z]), left_lane_points_num)

                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (left_point_down[1] - y1) - (y2 - y1) * (left_point_down[0] - x1)
                        if C < 0:
                            flag4 = '左下 线内'
                            if time_index >= start_time_index:
                                pass_flag4 = True

                    #right
                    agent_lane_right = agent_lane.right_edge
                    if agent_lane_right is not None:
                        rigt_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), agent_lane_right.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([right_point_up[0], right_point_up[1], z]), rigt_lane_points_num)
                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (right_point_up[1] - y1) - (y2 - y1) * (right_point_up[0] - x1)
                        if C > 0:
                            flag2 = '右上 线内'
                            if time_index >= start_time_index:
                                pass_flag2 = True

                    agent_lane_right = agent_lane.right_edge
                    if agent_lane_right is not None:
                        rigt_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), agent_lane_right.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([right_point_down[0], right_point_down[1], z]), rigt_lane_points_num)
                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (right_point_down[1] - y1) - (y2 - y1) * (right_point_down[0] - x1)
                        if C > 0:
                            flag3 = '右下 线内'
                            if time_index >= start_time_index:
                                pass_flag3 = True

                #print(f"timestamp: {timestamp}, laneid: {agent_lane.id}")
                #print(f"HV_front  flag1 = {flag1}, flag2 = {flag2}, flag3 = {flag3}, flag4 = {flag4}")
                if pass_flag1 == True and pass_flag2 == True and pass_flag3 == True and pass_flag4 == True:
                    end_time = timestamp
                    end_time_index = time_index
                    break
            extract_df.at[idx, 'lane_change_end_time'] = end_time
            extract_df.at[idx, 'lane_change_end_time_index'] = end_time_index
            extract_df.to_csv(interaction_idx_info, index=False)

        
        elif type == 1:
            for time_index, timestamp in enumerate(all_timesteps):
                flag1 = '左上 线外'
                flag2 = '右上 线外'
                flag3 = '右下 线外'
                flag4 = '左下 线外'

                pass_flag1 = False
                pass_flag2 = False
                pass_flag3 = False
                pass_flag4 = False
                ego = get_agent_state(column_dict, agents_states[ego_index, time_index, :])
                x, y, z, h = ego.x, ego.y, ego.z, ego.h
                #左上 右上
                corners = calculate_vehicle_bounding_box(x, y, length, width, h)
                left_point_up, right_point_up, right_point_down, left_point_down = corners[3], corners[2], corners[1], corners[0]

                ego_state = agents_states[ego_index, time_index, :]
                ego_lane = current_lane_for_ego(vec_map, column_dict, ego_state, length=length, width=width)
                #agent_lane = vec_map.get_closest_lane(np.array([x, y, z]))
                
                if ego_lane is not None:
                    #left
                    ego_lane_left = ego_lane.left_edge
                    if ego_lane_left is not None: 
                        left_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), ego_lane_left.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([left_point_up[0], left_point_up[1], z]), left_lane_points_num)

                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1

                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (left_point_up[1] - y1) - (y2 - y1) * (left_point_up[0] - x1)
                        if C < 0:
                            flag1 = '左上 线内'
                            if time_index >= start_time_index:
                                pass_flag1 = True
                    
                    if ego_lane_left is not None: 
                        left_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), ego_lane_left.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([left_point_down[0], left_point_down[1], z]), left_lane_points_num)

                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (left_point_down[1] - y1) - (y2 - y1) * (left_point_down[0] - x1)
                        if C < 0:
                            flag4 = '左下 线内'
                            if time_index >= start_time_index:
                                pass_flag4 = True

                    #right
                    ego_lane_right = ego_lane.right_edge
                    if ego_lane_right is not None:
                        rigt_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), ego_lane_right.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([right_point_up[0], right_point_up[1], z]), rigt_lane_points_num)
                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (right_point_up[1] - y1) - (y2 - y1) * (right_point_up[0] - x1)
                        if C > 0:
                            flag2 = '右上 线内'
                            if time_index >= start_time_index:
                                pass_flag2 = True

                    ego_lane_right = ego_lane.right_edge
                    if ego_lane_right is not None:
                        rigt_lane_points_num = find_closest_point_and_generate(np.array([x, y, z]), ego_lane_right.points)
                        closest_point, next_point = get_closest_points_for_lane_edge(np.array([right_point_down[0], right_point_down[1], z]), rigt_lane_points_num)
                        x1, y1 = closest_point[0], closest_point[1]
                        x2, y2 = next_point[0], next_point[1]
                        angle_diff = calculate_angle(x1, y1, x2, y2, h)
                        
                        if abs(angle_diff) > math.pi / 2:
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        A, B = np.array([x1, y1]), np.array([x2, y2])
                        C = (x2 - x1) * (right_point_down[1] - y1) - (y2 - y1) * (right_point_down[0] - x1)
                        if C > 0:
                            flag3 = '右下 线内'
                            if time_index >= start_time_index:
                                pass_flag3 = True

                #print(f"timestamp: {timestamp}, laneid: {agent_lane.id}")
                #print(f"AV_front  flag1 = {flag1}, flag2 = {flag2}, flag3 = {flag3}, flag4 = {flag4}")
                if pass_flag1 == True and pass_flag2 == True and pass_flag3 == True and pass_flag4 == True:
                    end_time = timestamp
                    end_time_index = time_index
                    break
            extract_df.at[idx, 'lane_change_end_time'] = end_time
            extract_df.at[idx, 'lane_change_end_time_index'] = end_time_index
            extract_df.to_csv(interaction_idx_info, index=False)

        
        if end_time_index != -1 and start_time_index != -1:
            delta_lats = []
            lane_change_type = 'None'
            for time_index, timestamp in enumerate(all_timesteps):
                if time_index < start_time_index or time_index > end_time_index:
                    continue
                
                ego = get_agent_state(column_dict, agents_states[ego_index, time_index, :])
                agent = get_agent_state(column_dict, agents_states[agent_index, time_index, :])

                delta_x = agent.x - ego.x
                delta_y = agent.y - ego.y

                delta_lat = -delta_x*math.sin(ego.h) + delta_y*math.cos(ego.h)
                #print(delta_lat)
                delta_lats.append(delta_lat)
            #print(delta_lats)
            if abs(delta_lats[0]) - abs(delta_lats[-1]) > 0:
                trend_flag = 'decreasing'
            else:
                trend_flag = 'increasing'
            #trend_flag = check_trend(delta_lats)
            #print(trend_flag)
            #print(f"proj = {proj}")

            type_flag = 'None'
            type_flag_init = 'None'
            if proj < -1.0:
                if type == 1:
                    if delta_lats[0] > 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, HV在ego后方直行, ego在HV前方向右侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'ego在HV右侧车道, HV直行, ego在其右前方, 向左变道到HV所在车道.'
                    elif delta_lats[0] < 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, HV在ego后方直行, ego在HV前方向左侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'ego在HV左侧车道, HV直行, ego在其左前方, 向右变道到HV所在车道.'
                
                elif type == 2:
                    if delta_lats[0] > 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, ego在HV前方直行, HV在ego后方向左侧车道变道.'

                        elif trend_flag == 'decreasing':
                            type_flag_init = 'HV在ego左侧车道, ego直行, HV在其左后方, 向右变道到ego所在车道.'
                    elif delta_lats[0] < 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, ego在HV前方直行, HV在ego后方向右侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'HV在ego右侧车道, ego直行, HV在其右后方, 向左变道到ego所在车道.'
            
            elif proj > 1.0:
                if type == 1:
                    if delta_lats[0] > 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, HV在ego前方直行, ego在HV后方向右侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'ego在HV右侧车道, HV直行, ego在其右后方, 向左变道到HV所在车道.'
                    elif delta_lats[0] < 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, HV在ego前方直行, ego在HV后方向左侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'ego在HV左侧车道, HV直行, ego在其左后方, 向右变道到HV所在车道.'
                
                elif type == 2:
                    if delta_lats[0] > 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, ego在HV后方直行, HV在ego前方向左侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'HV在ego左侧车道, ego直行, HV在其左前方, 向右变道到ego所在车道.'
                    elif delta_lats[0] < 0:
                        if trend_flag == 'increasing':
                            type_flag_init = 'ego与HV在同一车道上, ego在HV后方直行, HV在ego后方向右侧车道变道.'
                        elif trend_flag == 'decreasing':
                            type_flag_init = 'HV在ego右侧车道, ego直行, HV在其右前方, 向左变道到ego所在车道.'
                
            print(type_flag_init)

            extract_df.at[idx, 'type_flag_init'] = type_flag_init

        print(f"Updated CSV saved with lane change types at: {interaction_idx_info}")




if __name__ == "__main__":
    calculate_indicator(target_id=None)
