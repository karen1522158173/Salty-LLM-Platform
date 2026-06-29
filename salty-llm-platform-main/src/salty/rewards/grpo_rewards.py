"""GRPO 奖励函数

从旧 GRPO.py 抽出 calculate_grpo_rewards;按维度拆分子项以便配置 reward_weights。
"""
from collections import Counter
from typing import Dict


def reward_components(resp_text: str, target_text: str) -> Dict[str, float]:
    """返回奖励的各组成部分(用于日志和按 weight 加权)。

    返回的 key:
      - close_tag: 是否闭合 </think>
      - len_ok: 思考长度落在 [50, 400] 区间
      - len_overflow: 长度超 400 的惩罚
      - char_diversity: 极致重复字符惩罚
      - ngram_repeat: n-gram 重复惩罚
      - logic_chain: 第一步/第二步/所以|综上 顺序奖励
      - answer_match: 最终答案命中目标
      - no_close_tag: 没有闭合标签的重罚(直接终止)

    返回值含正负,trainer 处加权求和。
    """
    out: Dict[str, float] = {
        "close_tag": 0.0,
        "len_ok": 0.0,
        "len_overflow": 0.0,
        "char_diversity": 0.0,
        "ngram_repeat": 0.0,
        "logic_chain": 0.0,
        "answer_match": 0.0,
        "no_close_tag": 0.0,
    }

    if "</think>" not in resp_text:
        out["no_close_tag"] = -2.5
        return out

    out["close_tag"] = 1.0
    parts = resp_text.split("</think>")
    think_content = parts[0].replace("<think>", "").strip()
    ans_part = parts[1].strip()
    t_len = len(think_content)

    # A. 长度控制
    if 50 < t_len < 400:
        out["len_ok"] = 0.5
    elif t_len >= 400:
        penalty = ((t_len - 400) / 100) * 0.2
        out["len_overflow"] = -min(penalty, 1.5)

    # B. 反作弊
    if t_len > 0:
        if len(set(think_content)) / t_len < 0.15:
            out["char_diversity"] = -2.0

        if t_len > 20:
            grams = [think_content[i : i + 4] for i in range(t_len - 3)]
            _, count = Counter(grams).most_common(1)[0]
            if count > max(3, t_len * 0.05):
                out["ngram_repeat"] = -1.5

        step_1_idx = think_content.find("第一步")
        step_2_idx = think_content.find("第二步")
        conclusion_idx = max(think_content.find("所以"), think_content.find("综上"))

        logic_score = 0.0
        if step_1_idx != -1 and step_2_idx != -1 and step_1_idx < step_2_idx:
            logic_score += 0.3
        if conclusion_idx != -1 and conclusion_idx > len(think_content) * 0.5:
            logic_score += 0.2
        out["logic_chain"] = min(logic_score, 0.5)

    # C. 答案准确率
    if target_text in ans_part or (len(ans_part) > 5 and ans_part in target_text):
        out["answer_match"] = 2.0
    else:
        out["answer_match"] = -0.5

    return out


def calculate_grpo_reward(resp_text: str, target_text: str, weights: Dict[str, float] | None = None) -> float:
    """旧 calculate_grpo_rewards 的等价实现:weights 全为 1 时与原版完全一致。"""
    comps = reward_components(resp_text, target_text)
    if weights is None:
        return sum(comps.values())
    return sum(comps[k] * weights.get(k, 1.0) for k in comps)
