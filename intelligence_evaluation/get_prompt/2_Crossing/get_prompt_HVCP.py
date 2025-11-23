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

DELTA_T = 2
# 检查文件是否已存在（如果不存在，创建 CSV 并写入列名）

considertime = 5
starting_extension_time = 0.5
ending_extension_time = 2.5

length = 4.5
width = 1.8

base_dir = os.path.dirname(__file__)
idx_dir = os.path.join(base_dir, '..', 'index')
interaction_idx_info = os.path.join(idx_dir, 'waymo_CP_HV.csv')

FOLDER_CACHE_MAP = {
    'waymo_0-299': '/home/zjr/文档/InterHub_cache/waymo_0-299',
    'waymo_300-499': '/home/zjr/文档/InterHub_cache/waymo_300-499',
    'waymo_500-799': '/home/zjr/文档/InterHub_cache/waymo_500-799',
    'waymo_800-999': '/home/zjr/文档/InterHub_cache/waymo_800-999'
}
extract_df = pd.read_csv(interaction_idx_info)

results_dir = os.path.join(base_dir, '..', 'prompt_results')
results_file = os.path.join(results_dir, 'HVCP_prompt.csv')

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


def rad_to_chinese_direction(rad):
    deg = np.degrees(rad) % 360  # 弧度转角度，并映射到 [0, 360)

    def check_exact(target):
        return abs(deg - target) <= 1 or abs(deg - target + 360) <= 1

    if check_exact(0) or check_exact(360):
        return "正东"
    elif check_exact(90):
        return "正北"
    elif check_exact(180):
        return "正西"
    elif check_exact(270):
        return "正南"

    if 0 < deg < 90:
        return f"东偏北{round(deg)}度"
    elif 90 < deg < 180:
        return f"北偏西{round(deg - 90)}度"
    elif 180 < deg < 270:
        return f"西偏南{round(deg - 180)}度"
    elif 270 < deg < 360:
        return f"南偏东{round(deg - 270)}度"

datas = []
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
        """ if index > 10003:
            continue """

        turn_label = row['turn_label'].split('-')
        
        i=1
        key_agents = row['key_agents'].split(';')
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

#------------------------------------------------------------init_prompt------------------------------------------------------------
        ego_index = all_agents.index(ego_id)
        agent_index = all_agents.index(key_agent)

        ego_init_state = agents_states[ego_index, 0, :]
        AV_init_x, AV_init_y, AV_init_h = ego_init_state[column_dict['x']], ego_init_state[column_dict['y']], ego_init_state[column_dict['heading']]
        AV_init_v = ego_init_state[column_dict['vx']] * math.cos(AV_init_h) + ego_init_state[column_dict['vy']] * math.sin(AV_init_h)
        AV_init_a = ego_init_state[column_dict['ax']] * math.cos(AV_init_h) + ego_init_state[column_dict['ay']] * math.sin(AV_init_h)
        AV_init_direction = rad_to_chinese_direction(AV_init_h)

        agent_init_state = agents_states[agent_index, 0, :]
        HV_init_x, HV_init_y, HV_init_h = agent_init_state[column_dict['x']], agent_init_state[column_dict['y']], agent_init_state[column_dict['heading']]
        HV_init_v = agent_init_state[column_dict['vx']] * math.cos(HV_init_h) + agent_init_state[column_dict['vy']] * math.sin(HV_init_h)
        HV_init_a = agent_init_state[column_dict['ax']] * math.cos(HV_init_h) + agent_init_state[column_dict['ay']] * math.sin(HV_init_h)
        HV_init_direction = rad_to_chinese_direction(HV_init_h)

        print(agent_init_state[column_dict['ax']], agent_init_state[column_dict['ay']])

        turn_map = {
            'L': '左转',
            'S': '直行',
            'R': '右转',
        }

        mapping = dict(zip(key_agents, turn_label))
        AV_turn = turn_map.get(mapping.get('ego', ''), '')
        HV_turn = turn_map.get(mapping.get(key_agent, ''), '')

        priority_label = row['priority_label']
        if priority_label == 'ego':
            priority_agent = '在理论上具有优先行驶权的是AV'
        elif priority_label == 'equal':
            priority_agent = '在理论上两车具有相等的行驶权'
        else:
            priority_agent = '在理论上具有优先行驶权的是HV'

        prompt = f"""-----场景index: {index}. 当前场景为两车交叉交互, 且两车具有潜在冲突点, 该过程中ego{AV_turn}, HV{HV_turn}, {priority_agent}, 该场景持续时间为{interaction_end-interaction_start}秒.-----\n初始时刻, ego的位置为({AV_init_x:.3f}, {AV_init_y:.3f}), 以{AV_init_v:.3f} m/s的速度、{AV_init_a:.3f} m/s^2的加速度向{AV_init_direction}方向行驶; HV的位置为({HV_init_x:.3f}, {HV_init_y:.3f}), 以{HV_init_v:.3f} m/s的速度、{HV_init_a:.3f} m/s^2的加速度向{HV_init_direction}方向行驶;"""


#------------------------------------------------------------寻找冲突点------------------------------------------------------------
        for time_index, timestamp in enumerate(all_timesteps):

            collision_point, dis_AV, dis_HV, AV_speed, HV_speed, delta_AV, delta_HV = get_collision_point_with_dis_and_speed(column_dict, agents_states, ego_index, agent_index, time_index)
            
            if collision_point is not None:
                break

        prompt += f"""\n两车未来轨迹存在冲突点{collision_point}"""


#------------------------------------------------------------以相对于冲突点的距离定义交互开始时间------------------------------------------------------------
        start_time_index = -1
        start_time = -1
        for time_index, timestamp in enumerate(all_timesteps):
            ego_state = agents_states[ego_index, time_index, :]
            AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
            AV_h = ego_state[column_dict['heading']]
            AV_v = ego_state[column_dict['vx']] * math.cos(AV_h) + ego_state[column_dict['vy']] * math.sin(AV_h)
            AV_a = ego_state[column_dict['ax']] * math.cos(AV_h) + ego_state[column_dict['ay']] * math.sin(AV_h)
            AV_dis = collision_point.distance(Point(AV_x, AV_y))

            AV_direction = rad_to_chinese_direction(AV_h)

            agent_state = agents_states[agent_index, time_index, :]
            HV_x, HV_y = agent_state[column_dict['x']], agent_state[column_dict['y']]
            HV_h = agent_state[column_dict['heading']]
            HV_v = agent_state[column_dict['vx']] * math.cos(HV_h) + agent_state[column_dict['vy']] * math.sin(HV_h)
            HV_a = agent_state[column_dict['ax']] * math.cos(HV_h) + agent_state[column_dict['ay']] * math.sin(HV_h)
            HV_dis = collision_point.distance(Point(HV_x, HV_y))

            HV_direction = rad_to_chinese_direction(HV_h)

            dis_threshold = 15 #更改阈值以更改判定交互开始的方式
            if AV_dis is not None and HV_dis is not None:
                if AV_dis <= dis_threshold or HV_dis <= dis_threshold:
                    prompt += f"""\n第{time_index}秒时，两车进入交互范围. 该时刻, ego距离冲突点{AV_dis:.3f}米, 速度{AV_v:.3f} m/s, 并以{AV_a:.3f} m/s^2的加速度朝着{AV_direction}方向行驶; HV距离冲突点{HV_dis:.3f}米, 速度{HV_v:.3f} m/s, 并以{HV_a:.3f} m/s^2的加速度朝着{HV_direction}方向行驶."""
                    start_time = timestamp
                    start_time_index = time_index
                    break



#------------------------------------------------------------定义结束时间和方式: 1. 有一辆车通过了冲突点; 2. 交互时间已过------------------------------------------------------------
        ego_end_state = agents_states[ego_index, interaction_end-interaction_start-1, :]
        AV_end_x, AV_end_y, AV_end_h = ego_end_state[column_dict['x']], ego_end_state[column_dict['y']], ego_end_state[column_dict['heading']]
        AV_end_dis = collision_point.distance(Point(AV_end_x, AV_end_y))

        AV_end_vx, AV_end_vy = ego_end_state[column_dict['vx']], ego_end_state[column_dict['vy']]
        AV_end_v = AV_end_vx*math.cos(AV_end_h) + AV_end_vy*math.sin(AV_end_h)

        agent_end_state = agents_states[agent_index, interaction_end-interaction_start-1, :]
        HV_end_x, HV_end_y, HV_end_h = agent_end_state[column_dict['x']], agent_end_state[column_dict['y']], agent_end_state[column_dict['heading']]
        HV_end_dis = collision_point.distance(Point(HV_end_x, HV_end_y))

        HV_end_vx, HV_end_vy = agent_end_state[column_dict['vx']], agent_end_state[column_dict['vy']]
        HV_end_v = HV_end_vx*math.cos(HV_end_h) + HV_end_vy*math.sin(HV_end_h)
    
        #如果到片段最终时间也没有车辆通过冲突点, 则结束
        if AV_end_dis >= HV_end_dis:
            nearest_agent = f'HV距离冲突点最近, 为{HV_end_dis:.2f}米, AV距离冲突点{AV_end_dis:.2f}米'

            now_PET = (AV_end_dis-length/2-width/2)/AV_end_v - (HV_end_dis+width/2)/HV_end_v

        else:
            nearest_agent = f'AV距离冲突点最近, 为{AV_end_dis:.2f}米, HV距离冲突点{HV_end_dis:.2f}米'

            now_PET = (HV_end_dis-length/2-width/2)/HV_end_v - (AV_end_dis+width/2)/AV_end_v

        sentence = f"""\n第{interaction_end - interaction_start - 1}秒时, 该交互片段结束, {nearest_agent}."""

        #如果有一辆车率先通过了冲突点，则交互结束
        AV_vs = []
        HV_vs = []
        AV_as = []
        HV_as = []
        AV_dire = []
        HV_dire = []
        PETs = []
        AV_pre_x, AV_pre_y = AV_init_x, AV_init_y
        HV_pre_x, HV_pre_y = HV_init_x, HV_init_y

        end_time_index = 999
        end_time = 999
        for time_index, timestamp in enumerate(all_timesteps):
            if timestamp <= start_time:
                continue
            ego_state = agents_states[ego_index, time_index, :]
            AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
            AV_h = ego_state[column_dict['heading']]
            AV_v = ego_state[column_dict['vx']] * math.cos(AV_h) + ego_state[column_dict['vy']] * math.sin(AV_h)
            AV_a = ego_state[column_dict['ax']] * math.cos(AV_h) + ego_state[column_dict['ay']] * math.sin(AV_h)
            AV_dis = collision_point.distance(Point(AV_x, AV_y))
            AV_direction = rad_to_chinese_direction(AV_h)

            AV_vs.append(float(round(AV_v, 2)))
            AV_as.append(float(round(AV_a, 2)))
            AV_dire.append(AV_direction)

            agent_state = agents_states[agent_index, time_index, :]
            HV_x, HV_y = agent_state[column_dict['x']], agent_state[column_dict['y']]
            HV_h = agent_state[column_dict['heading']]
            HV_v = agent_state[column_dict['vx']] * math.cos(HV_h) + agent_state[column_dict['vy']] * math.sin(HV_h)
            HV_a = agent_state[column_dict['ax']] * math.cos(HV_h) + agent_state[column_dict['ay']] * math.sin(HV_h)
            HV_dis = collision_point.distance(Point(HV_x, HV_y))
            HV_direction = rad_to_chinese_direction(HV_h)

            HV_vs.append(float(round(HV_v, 2)))
            HV_as.append(float(round(HV_a, 2)))
            HV_dire.append(HV_direction)

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
                    sentence = f"""\n第{time_index}秒后, AV率先通过了冲突点, 交互结束, 此时HV距离冲突点{HV_dis:.2f}米, 预计将在{((HV_dis-length/2-width/2)/HV_v):.2f}秒后到达冲突点."""
                    now_PET = (HV_dis-length/2-width/2)/HV_v
                    end_time_index = time_index
                    end_time = timestamp
                    break

                HV_v1 = np.array([collision_point.x-HV_pre_x, collision_point.y-HV_pre_y])
                HV_v2 = np.array([collision_point.x-HV_x, collision_point.y-HV_y])
                if np.dot(HV_v1, HV_v2) < 0:
                    sentence = f"""\n第{time_index}秒后, HV率先通过了冲突点, 交互结束, 此时AV距离冲突点{AV_dis:.2f}米, 预计将在{((AV_dis-length/2-width/2)/AV_v):.2f}秒后到达冲突点."""
                    now_PET = (AV_dis-length/2-width/2)/AV_v

                    end_time_index = time_index
                    end_time = timestamp
                    break

                AV_pre_x, AV_pre_y = AV_x, AV_y
                HV_pre_x, HV_pre_y = HV_x, HV_y

        min_PET = min(PETs)
        prompt += sentence


        prompt += f"""该交互过程中, 以Collision Point为冲突区, 计算ego与HV的最小PET(PET, 后侵入时间, 一车离开冲突区与另一车到达冲突区的时间差)为: {min_PET} s. """
        prompt += f"""从交互开始之后:\n -ego的速度序列为: {AV_vs} m/s\n -ego的加速度序列为: {AV_as} m/s^2\n -ego的行驶方向序列为: {AV_dire};"""

        AV_v_lats = []
        AV_a_lats = []
        AV_yawrates = []
        for time_index, timestamp in enumerate(all_timesteps):
            if time_index > end_time_index:
                break
            ego_state = agents_states[ego_index, time_index, :]
            AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
            AV_h = ego_state[column_dict['heading']]
            AV_v_lat = -ego_state[column_dict['vx']] * math.sin(AV_h) + ego_state[column_dict['vy']] * math.cos(AV_h)#横向
            AV_a_lat = -ego_state[column_dict['ax']] * math.sin(AV_h) + ego_state[column_dict['ay']] * math.cos(AV_h)#横向
            AV_v_lat = float(round((AV_v_lat),3))
            AV_a_lat = float(round((AV_a_lat),3))
            
            AV_yawrate = (ego_state[column_dict['vx']]*ego_state[column_dict['ay']] - ego_state[column_dict['vy']]*ego_state[column_dict['ax']]) / (ego_state[column_dict['vx']]**2 + ego_state[column_dict['vy']]**2)
            AV_yawrate = float(round((AV_yawrate),3))

            AV_v_lats.append(AV_v_lat)
            AV_a_lats.append(AV_a_lat)
            AV_yawrates.append(AV_yawrate)

        prompt += f"""\n从0时刻到交互结束时刻(0~{end_time_index}秒)过程中:\n -ego的横向速度(垂直于车头方向)序列为: {AV_v_lats} m/s;\n -ego的横向加速度序列为: {AV_a_lats} m/s^2;\n -ego的横摆角速度为: {AV_yawrates} rad/s"""

        print(prompt)
        data = {'indexs': index,
                'prompt': prompt,
                'PET_min': min_PET,
                'now_PET': now_PET,}
    
        datas.append(data)
        
    df = pd.DataFrame(datas)


    # 写入 CSV 文件（不写索引列）
    df.to_csv(results_file, index=False)

if __name__ == "__main__":
    target_id = None
    traj_folder = "base_data/waymo_train"
    calculate_indicator(target_id, traj_folder)
