"""
该脚本用于构建"AV在HV后切入"场景的prompt
最终的prompt将保存在csv文件中, 该文件中第一列为场景索引号, 第二列为prompt, 第三列为ttc
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

base_dir = os.path.dirname(__file__)
idx_dir = os.path.join(base_dir, '..', 'index')
interaction_idx_info = os.path.join(idx_dir, 'waymo_MP_AVback.csv')

FOLDER_CACHE_MAP = {
    'waymo_0-299': '/home/zjr/文档/InterHub_cache/waymo_0-299',
    'waymo_300-499': '/home/zjr/文档/InterHub_cache/waymo_300-499',
    'waymo_500-799': '/home/zjr/文档/InterHub_cache/waymo_500-799',
    'waymo_800-999': '/home/zjr/文档/InterHub_cache/waymo_800-999'
}
extract_df = pd.read_csv(interaction_idx_info)

results_dir = os.path.join(base_dir, '..', 'prompt_results')
results_file = os.path.join(results_dir, 'AVback_prompt.csv')

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




def load_csv_by_index(index, data_dir="base_data/waymo_train"):
    index_str = str(index) + "_"
    
    # 遍历文件寻找匹配的index
    for filename in os.listdir(data_dir):
        if filename.startswith(index_str) and filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            return df, filename
    
    # 没找到就返回None
    return None, None


datas = []
def calculate_indicator():
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
        """ if index > 10:
            continue """
        """ if index != 50:
            continue """
        lane_change_type = row['lane_change_type']
        lane_change_time_index = int(row['lane_change_time_index'])
        lane_change_end_time_index = int(row['lane_change_end_time_index'])
        type_flag_init = row['type_flag_init']

        if lane_change_type != 'AV_back':
            continue

        """ if index > 53:
            continue """

        #在规定时间内HV没有完成换道
        if lane_change_end_time_index == -1:
            continue
        
        for agent in key_agents:
            if agent != 'ego':
                key_agent = agent

        flag2 = False
        for agent in key_agents:
            if agent == 'ego':
                flag2 = True
        if flag2 == False:
            continue
        
        for agent in key_agents:
            if agent != 'ego':
                key_agent = agent

        ego_id = next((agent for agent in interact_ids if 'ego' in agent), None)
        agent_ids = [a for a in interact_ids if a != ego_id]

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

        dis_init = math.sqrt((AV_init_x - HV_init_x)**2 + (AV_init_y - HV_init_y)**2)

        prompt = f"""-----场景index: {index}. 当前交互场景为: {type_flag_init} 在该过程中, 我们需要着重关注ego在变道时自身的状态, 以及与前车HV的交互情况-----\n初始时刻, ego的位置为({AV_init_x:.3f}, {AV_init_y:.3f}), 以{AV_init_v:.3f} m/s的速度、{AV_init_a:.3f} m/s^2的加速度向{AV_init_direction}方向行驶; HV的位置为({HV_init_x:.3f}, {HV_init_y:.3f}), 以{HV_init_v:.3f} m/s的速度、{HV_init_a:.3f} m/s^2的加速度向{HV_init_direction}方向行驶; 两车相距(车辆中心点相对距离){dis_init:.2f} m."""

#------------------------------------------------------------交互开始时间------------------------------------------------------------

        ego_state = agents_states[ego_index, lane_change_time_index, :]
        AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
        AV_h = ego_state[column_dict['heading']]
        AV_v = ego_state[column_dict['vx']] * math.cos(AV_h) + ego_state[column_dict['vy']] * math.sin(AV_h)
        AV_a = ego_state[column_dict['ax']] * math.cos(AV_h) + ego_state[column_dict['ay']] * math.sin(AV_h)
        AV_direction = rad_to_chinese_direction(AV_h)

        agent_state = agents_states[agent_index, lane_change_time_index, :]
        HV_x, HV_y = agent_state[column_dict['x']], agent_state[column_dict['y']]
        HV_h = agent_state[column_dict['heading']]
        HV_v = agent_state[column_dict['vx']] * math.cos(HV_h) + agent_state[column_dict['vy']] * math.sin(HV_h)
        HV_a = agent_state[column_dict['ax']] * math.cos(HV_h) + agent_state[column_dict['ay']] * math.sin(HV_h)
        HV_direction = rad_to_chinese_direction(HV_h)

        ttc_value = calculate_ttc_with_one_agent_in_currenttime(column_dict, agents_states, ego_index, agent_index, lane_change_time_index)

        dis = math.sqrt((AV_x - HV_x)**2 + (AV_y - HV_y)**2)

        prompt += f"""\n第{lane_change_time_index}秒时, ego的部分车身跨过其当前所在车道, 预示其即将进行变道行为. 该时刻, ego位置为({AV_x:.3f}, {AV_y:.3f}), 以{AV_v:.3f} m/s的速度, 朝着{AV_direction}方向进行变道操作; HV当前时刻速度为{HV_v:.3f} m/s, 加速度为{HV_a:.3f} m/s^2, 朝着{HV_direction}方向行驶, 此时, 两车相距{dis:.2f} m, 根据ego与HV当前状态可以计算AV与HV的TTC为{ttc_value:.3f}"""
        #prompt += f"""\n第{lane_change_time_index}秒时, HV的部分车身跨过其当前所在车道, 预示其即将进行变道行为. 该时刻, HV位置为({HV_x:.3f}, {HV_y:.3f}), 以{HV_v:.3f} m/s的速度, 朝着{HV_direction}方向进行变道操作; AV当前时刻速度为{AV_v:.3f} m/s, 加速度为{AV_a:.3f} m/s^2, 朝着{AV_direction}方向行驶"""

#------------------------------------------------------------定义结束时间和方式: 1. cut-in车全部车身完成换道; 2. 交互时间已过------------------------------------------------------------
        ego_end_state = agents_states[ego_index, lane_change_end_time_index, :]
        AV_end_x, AV_end_y, AV_end_h = ego_end_state[column_dict['x']], ego_end_state[column_dict['y']], ego_end_state[column_dict['heading']]
        AV_end_v = ego_end_state[column_dict['vx']] * math.cos(AV_end_h) + ego_end_state[column_dict['vy']] * math.sin(AV_end_h)
        AV_end_a = ego_end_state[column_dict['ax']] * math.cos(AV_end_h) + ego_end_state[column_dict['ay']] * math.sin(AV_end_h)
        AV_end_direction = rad_to_chinese_direction(AV_end_h)


        agent_end_state = agents_states[agent_index, lane_change_end_time_index, :]
        HV_end_x, HV_end_y, HV_end_h = agent_end_state[column_dict['x']], agent_end_state[column_dict['y']], agent_end_state[column_dict['heading']]
        HV_end_v = agent_end_state[column_dict['vx']] * math.cos(HV_end_h) + agent_end_state[column_dict['vy']] * math.sin(HV_end_h)
        HV_end_a = agent_end_state[column_dict['ax']] * math.cos(HV_end_h) + agent_end_state[column_dict['ay']] * math.sin(HV_end_h)
        HV_end_direction = rad_to_chinese_direction(HV_end_h)

        if type_flag_init == "ego与HV在同一车道上, HV在ego前方直行, ego在HV后方向右侧车道变道.":
            prompt += f"""\n第{lane_change_end_time_index}秒时, ego全部车身均位于右侧车道, 预示着其完成变道操作. ego进行变道的持续时间为{lane_change_end_time_index - lane_change_time_index}秒. """
        elif type_flag_init == "ego与HV在同一车道上, HV在ego前方直行, ego在HV后方向左侧车道变道.":
            prompt += f"""\n第{lane_change_end_time_index}秒时, ego全部车身均位于左侧车道, 预示着其完成变道操作. ego进行变道的持续时间为{lane_change_end_time_index - lane_change_time_index}秒. """
        else:
            prompt += f"""\n第{lane_change_end_time_index}秒时, ego全部车身均位于HV所在车道, 预示着其完成变道操作. ego进行变道的持续时间为{lane_change_end_time_index - lane_change_time_index}秒. """

        #如果有一辆车率先通过了冲突点，则交互结束
        AV_vs = []
        HV_vs = []
        AV_as = []
        HV_as = []
        AV_dire = []
        HV_dire = []
        ttc_values = []
        ttc_flag = False
        for time_index, timestamp in enumerate(all_timesteps):
            if time_index <= lane_change_time_index:
                continue

            if time_index > lane_change_end_time_index:
                break
            ego_state = agents_states[ego_index, time_index, :]
            AV_x, AV_y = ego_state[column_dict['x']], ego_state[column_dict['y']]
            AV_h = ego_state[column_dict['heading']]
            AV_v = ego_state[column_dict['vx']] * math.cos(AV_h) + ego_state[column_dict['vy']] * math.sin(AV_h)
            AV_a = ego_state[column_dict['ax']] * math.cos(AV_h) + ego_state[column_dict['ay']] * math.sin(AV_h)
            AV_direction = rad_to_chinese_direction(AV_h)

            AV_vs.append(float(round(AV_v, 2)))
            AV_as.append(float(round(AV_a, 2)))
            AV_dire.append(AV_direction)

            agent_state = agents_states[agent_index, time_index, :]
            HV_x, HV_y = agent_state[column_dict['x']], agent_state[column_dict['y']]
            HV_h = agent_state[column_dict['heading']]
            HV_v = agent_state[column_dict['vx']] * math.cos(HV_h) + agent_state[column_dict['vy']] * math.sin(HV_h)
            HV_a = agent_state[column_dict['ax']] * math.cos(HV_h) + agent_state[column_dict['ay']] * math.sin(HV_h)
            HV_direction = rad_to_chinese_direction(HV_h)

            HV_vs.append(float(round(HV_v, 2)))
            HV_as.append(float(round(HV_a, 2)))
            HV_dire.append(HV_direction)

        for time_index, timestamp in enumerate(all_timesteps):
            if time_index > lane_change_end_time_index:
                break
            ttc_value = calculate_ttc_with_one_agent_in_currenttime(column_dict, agents_states, ego_index, agent_index, time_index)
            if ttc_value != np.inf:
                ttc_flag = True
            ttc_values.append(float(round(ttc_value,3)))

        print(ttc_values)
        if ttc_flag == False:
            ttc = np.inf
        else:
            ttc = min(ttc_values)
        
        print(ttc)

        prompt += f"""从ego进行变道行为开始之后:\n -ego的速度变化序列为: {AV_vs} m/s;\n -ego的加速度变化序列为: {AV_as} m/s^2;\n -ego的行驶方向变化序列为: {AV_dire};"""
        prompt += f"""\n从0时刻到ego变道结束这一过程中, 其与前车HV的TTC值(该TTC值的计算考虑了车身边缘碰撞)变化序列为{ttc_values} s"""

        i = index

        AV_v_lats = []
        AV_a_lats = []
        AV_yawrates = []
        for time_index, timestamp in enumerate(all_timesteps):
            if time_index > lane_change_end_time_index:
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

        prompt += f"""\n从0时刻到ego变道结束这一过程中, 为考虑ego的舒适性我们列出了:\n -ego的横向(垂直于车头方向)速度序列为: {AV_v_lats} m/s;\n -ego的横向加速度序列为: {AV_a_lats} m/s^2;\n -ego的横摆角速度为: {AV_yawrates} rad/s"""

        print(prompt)

        data = {'indexs': i,
                'prompt': prompt,
                'ttc_min': ttc}
    
        datas.append(data)

    df = pd.DataFrame(datas)


    # 写入 CSV 文件（不写索引列）
    df.to_csv(results_file, index=False)


if __name__ == "__main__":

    calculate_indicator()
