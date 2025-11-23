import os
import pandas as pd
from openai import OpenAI

# ==============================
# 多 API Key 配置
# ==============================
API_KEYS = [
'sk-OGEcvMP5ytY3mZ3UDfFb56C7E8Ae4aEf9dB3E0FcCf6e8bBf',
'sk-F5PyQv5XOuaPoL248cFfE2Ac11F24cFbBc61E0E7294eBb49',
'sk-JpA90QCgUwhZncSd1523B0B7A2854d4983E1D933283eC2Bf',
'sk-WYuluBgM76JaV83n57Ac55B27e284d5f97D6797d75989aF5',
'sk-gkaVDX46F5UMksEg02EaCa3024C04eA2B1140bCd60B6B2B6',
'sk-oIYNzhUdSXu6eR5q304280E730D543A09f9eC3196bB5AbF1',
'sk-o864zUTjqxsboiCDAdC0C296DcC8471a91AfA19e350eCe6e',
'sk-Y55F8U3yDZHlWwZK36Fb681820694531Ac2e8d65E82611D2',
'sk-1DfFQDrvdDqhC4XmA196Fe515cF444899147C5A29d7a67Fb',
'sk-ikH1fcxMBBhxF52P577dDdB1F583487a86Df6bFaCd101079',
'sk-EloV4mxhKYH64HHXD0F8C63a48214b76AeFb47322e500fAc',
'sk-OHbHkeverRpRXF9g69Ab890c7fBc44Ad84758b250f896516',
'sk-QVwQUCYQPqWcEnZe0a377e3665E64d278d2625B99dB571Cf',
'sk-ZnDGLvUyhktdPfQD372c7325Ee564c548dE85504A4D6F61f',
'sk-BFmgl132N2m2yJPU0bF544D98035420a9114B5F6F7C90e39',
'sk-19DgvMXWDJzApxjn94E87b817d8c45FeA2019074FeAeA40e',
'sk-jUWjj4GwcR19cbIDE28e8867969a490590Eb35E288Dc8bF1',
    # 可以继续添加更多的 API_KEY
]

# 设置 API 基础 URL（根据你第一段的配置）
os.environ["OPENAI_BASE_URL"] = "http://66.206.9.230:4000/v1"
os.environ["OPENAI_API_KEY"] = API_KEYS[0]

# 当前使用的 key 索引
current_api_key_idx = 0


# ==============================
# 获取客户端
# ==============================
def get_openai_client():
    global current_api_key_idx
    api_key = API_KEYS[current_api_key_idx]
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI()


# ==============================
# GPT 调用函数（带自动切换 key）
# ==============================
def extract_score_with_gpt(text):
    """
    使用 GPT-4o 从自动驾驶评价文本中提取得分（仅保留一位小数）。
    """
    global current_api_key_idx

    try:
        client = get_openai_client()
        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            temperature=0.0,
            max_tokens=50,
            messages=[
                {
                    "role": "system",
                    "content": "你是一名文本信息抽取专家，任务是从描述中精确提取最终分数。",
                },
                {
                    "role": "user",
                    "content": f"""
从下面的自动驾驶智能度评价文本中，提取出最终的智能度综合得分。
要求：
1. 只返回一个保留一位小数的浮点数（float）。
2. 不要返回任何解释或文字。
3. 如果文本中的得分表达形式是“xxx/100”，则直接取 xxx 作为结果（不要转化为百分比）。
【文本】：
{text}
                    """,
                },
            ],
        )

        # 提取模型回复
        result = completion.choices[0].message.content.strip()

        # 基础验证
        if not result or not any(c.isdigit() for c in result):
            raise ValueError("无有效数字结果")

        return result

    except Exception as e:
        print(f"Error with current API key (Index {current_api_key_idx}): {e}")
        current_api_key_idx = (current_api_key_idx + 1) % len(API_KEYS)
        print(f"Switching to API key (Index {current_api_key_idx}) ...")
        return extract_score_with_gpt(text)  # 递归重试


# ==============================
# 主流程
# ==============================
import os
def calculate_indicator(target_id=None):
    file_dir = os.path.join(os.getcwd(), "evaluation_results")
    prompt_info = f'{file_dir}/evaluation_results4.csv'
    extract_df = pd.read_csv(prompt_info)

    results = []

    for rank, (idx, row) in enumerate(extract_df.iterrows(), start=1):
        index = row['index']
        response_text = row['response']

        """ if index >= 7823:
            continue """

        print(f"\n\nindex: {index}")
        print(f"------------------输入------------------")
        print(response_text)
        print(f"------------------结束输入------------------")

        score = extract_score_with_gpt(response_text)

        print(f"\n------------------提取结果------------------")
        print(score)
        print(f"------------------结束结果------------------")

        results.append({"index": index, "score": score})

        # 每 5 次保存一次
        if len(results) % 5 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(
                "evaluation_score4.csv",
                mode="a",
                header=not os.path.exists("evaluation_score4.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            results = []

    # 保存剩余结果
    if results:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(
            "evaluation_score4.csv",
            mode="a",
            header=not os.path.exists("evaluation_score4.csv"),
            index=False,
            encoding="utf-8-sig",
        )


# ==============================
# 程序入口
# ==============================
if __name__ == "__main__":
    calculate_indicator()
