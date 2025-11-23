import os
import pandas as pd
from openai import OpenAI

# 设置多个 API 密钥
API_KEYS = [


    # 可以继续添加更多的 API_KEY
]

# 设置 OpenAI API 基础 URL
os.environ["OPENAI_BASE_URL"] = "http://66.206.9.230:4000/v1"

os.environ["OPENAI_API_KEY"] = "sk-fa9qC4jhF2vha2Ly803a788c17A04b698996B112D8Ba50Bc"
# 初始 API_KEY 索引
current_api_key_idx = 0

# 设置 OpenAI 客户端
client = OpenAI()

interaction_idx_info = 'all_prompt.csv'
extract_df = pd.read_csv(interaction_idx_info)

def get_openai_client():
    """
    获取当前 API 客户端实例，使用当前的 API_KEY。
    """
    global current_api_key_idx
    api_key = API_KEYS[current_api_key_idx]
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()

def evaluate_with_gpt(description):
    """
    让 GPT 评分 (适配新版 OpenAI API)，自动切换 API_KEY。
    """
    global current_api_key_idx
    
    # 这里我们捕捉 API 调用时的异常（比如密钥用尽的错误）
    try:
        client = get_openai_client()
        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            max_tokens=1000,
            temperature=0.1,
            top_p=1,
            messages=[
                {"role": "system", "content": "你是自动驾驶智能度评估专家，请基于下面自动驾驶车辆与其他车辆交互数据对自动驾驶的智能度进行评价。"},
                {"role": "user", "content": description},
            ]
        )
        
        if getattr(completion.choices[0].message, 'content', None):
            content = completion.choices[0].message.content
        else:
            print('error_wait_2s')
            return None

        return content

    except Exception as e:
        print(f"Error with current API_KEY (Index {current_api_key_idx}): {e}")
        # 切换到下一个 API_KEY
        current_api_key_idx = (current_api_key_idx + 1) % len(API_KEYS)
        print(f"Switching to API_KEY (Index {current_api_key_idx})")
        return evaluate_with_gpt(description)  # 重试

def calculate_indicator(target_id):
    results = []  # 用于存储评分结果

    for rank, (idx, row) in enumerate(extract_df.iterrows(), start=1):
        index = row['indexs']
        prompt = row['prompt']

        if index <= 6849:
            continue

        input = f'''你是一名自动驾驶行为理解与评价专家, 请基于【自动驾驶评价准则与知识】, 结合你自己的推理与分析, 进行合理的评价...'''

        input += f'''【交互测试内容】'''
        input += prompt
        input += f'''【自动驾驶评价准则与知识】(变量落在不同范围, 该维度的得分应有所差别)
--TTC范围: [0, 1.5s] 很危险, 可能碰撞; [1.5s, 3.0s] 存在潜在碰撞风险, 需要减速或刹车; [3.0s, 5.0s] 较为安全, 但接近时建议减速准备; [> 5.0s] 安全状态, 风险低, 越高的TTC说明越安全, 如果为inf则表示当前时刻没有碰撞风险。
--PET范围: [0, 0.8s] 高危险, 可能碰撞; [0.8s, 1.5s] 中等风险, 可能需刹车或转向规避; [1.5s, 3.0s] 可接受风险, 大多数情况下可控; [> 3.0s] 安全, 风险极低。
--纵向加速度(a_long): [0, 0.3] 舒适性很好, 乘客无不适感; [0.30, 1.23] 可接受, 轻微不适感; [1.23, 2.12] 不适, 急刹车或加速会引起不适; [> 2.12] 舒适性差, 乘客极差体验, 安全性存在隐患。
--横摆角速度(yaw_rate): [0, 0.1] 车辆平稳, 乘客无不适; [0.1, 0.23] 轻微转弯或变道, 可能有轻微不适感; [0.23, 0.4] 急转弯或变道, 乘客可能有晕车感; [> 0.4] 紧急转向或失控, 乘客极易不适, 安全性差。
--横向加速度(a_lat): [0, 0.5] 行驶平稳, 舒适性好; [0.5, 1.0] 轻微转弯或变道, 乘客不适轻微; [1.0, 1.5] 快速转弯或变道, 舒适性下降; [1.5, 2.5] 不适, 存在安全隐患; [> 2.5] 严重不适, 车辆稳定性受威胁, 安全性风险高。
--横向速度(v_lat): [0, 0.5] 行驶平稳, 舒适性好; [0.5, 1.5] 小幅侧滑或变道, 舒适性下降; [1.5, 2.5] 打滑、甩尾等异常, 舒适性差, 安全性威胁; [> 2.5] 严重侧滑, 车辆失控, 舒适性极差, 安全性受威胁。
--如果车辆行驶过程当中, 出现加速度序列变化波动较大, 或加速度数值绝对值特别大, 或者加速/减速频繁, 加速度正、负之间来回振荡的情况, 或者横摆角速度持续较高、朝向角变化明显, 舒适性较差, 并且也会导致行驶效率较低、油耗增加, 且带来一定的安全隐患.
--若车辆速度曲线在短时间内频繁上下振荡, 说明其加减速控制策略不稳定, 影响乘客体验, 可能导致能耗增加和后车追尾风险上升.
--若车辆跟驰或变道过程中, TTC值在很安全的范围中(比如大于5.0或inf), 但是车速却持续很低(比如一直保持在10m/s左右或以内), 说明车辆在效率性智能度上的表现有待提高'''

        input += f'''\n评价分析应全面言简意赅, 注意控制字数, 回复格式为一整段文字, 无需分点, 并且合理地给出最终的综合得分, 该最终得分应在[0, 10]范围内, 共11个等级.'''

        print(f'\n\nindex: {index}')
        print(f'------------------input------------------')
        print(input)
        print(f'------------------end input------------------')

        response = evaluate_with_gpt(input)
        
        # 将评分结果保存到列表
        if response:
            results.append([index, response])

        # 每10次评价后自动保存结果
        if len(results) % 2 == 0:
            temp_df = pd.DataFrame(results, columns=["index", "response"])
            temp_df.to_csv("evaluation_results41.csv", mode='a', header=not os.path.exists("evaluation_results41.csv"), index=False, encoding='utf-8-sig')

            # 清空当前存储的结果列表
            results = []

        print(f'\n\n------------------response------------------')
        print(response)
        print(f'------------------end response------------------')

    # 如果剩余的评价未保存，保存它们
    if results:
        temp_df = pd.DataFrame(results, columns=["index", "response"])
        temp_df.to_csv("evaluation_results41.csv", mode='a', header=not os.path.exists("evaluation_results41.csv"), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    target_id = None
    calculate_indicator(target_id)