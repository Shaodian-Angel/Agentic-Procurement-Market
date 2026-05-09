import random
import re
import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from collections import Counter

matplotlib.use('TkAgg')
import seaborn as sns
from utils.gpt_inference import GPTInference


# ==========================================
# 1. 主动型卖方系统 (Active Seller - FSM)
# ==========================================
class ActiveSeller:
    """
    基于有限状态机 (FSM) 的主动卖方
    状态定义:
    - Baseline: 初始报价 = reservation_price + markup
    - PushBack: 检测到 buyer concession rate < lambda_threshold，加价
    - Concession: 连续两轮 buyer 提价 > delta，降价
    - RewardTrust: 满足信任条件，触发折扣
    """

    def __init__(self, seller_id, seller_type,
                 lambda_threshold=0.01,  # [优化] 从 0.08 降至 0.02。减少对微小让步的过度反应 # 0.01-0.02
                 delta_ask=0.1,  # [优化] 从 7 降至 5。PushBack 时的加价幅度减小，避免价格剧烈震荡 # 改为百分比更好，10%或更大会更好
                 gamma=0.05,  # 让步幅度保持不变 # 改为百分比，和delta_ask之间的差距更大一些？
                 discount_rate=0.1):  # 折扣率保持不变 # 更大一些，0.1-0.08
        self.seller_id = seller_id
        self.type = seller_type  # "hard" or "flexible"

        # 基础定价参数
        self.reservation_price = 150 if seller_type == "hard" else 110
        self.base_markup = 20
        self.current_ask = self.reservation_price + self.base_markup

        # FSM 状态参数
        self.state = 'Baseline'
        self.lambda_threshold = lambda_threshold
        self.delta_ask = delta_ask
        self.gamma = gamma
        self.discount_rate = discount_rate

        # 历史观测信号
        self.prev_bid = None
        self.bid_history = []
        self.interaction_count = 0
        self.reward_active = False  # 标记当前是否处于 RewardTrust 的折扣生效中

    def _calculate_concession_rate(self, current_bid):
        if self.prev_bid is None or self.prev_bid == 0:
            return 1.0
        # 计算买方让步率：(当前出价 - 上次出价) / (上次出价)
        return (current_bid - self.prev_bid) / self.prev_bid

    def _check_reward_trust_condition(self, current_bid):
        # 条件：过去3轮中至少2次出价 >= ask-5 且 interaction_count >= 2
        if self.interaction_count < 2:
            return False

        recent_bids = self.bid_history[-3:]
        threshold_price = self.current_ask - 5
        high_bids_count = sum(1 for b in recent_bids if b >= threshold_price)
        return high_bids_count >= 2

    def _transition_state(self, current_bid):
        prev_state = self.state
        concession_rate = self._calculate_concession_rate(current_bid)

        # 1. 检查是否触发 RewardTrust (优先级最高)
        if self._check_reward_trust_condition(current_bid):
            self.state = 'RewardTrust'
            self.reward_active = True
        else:
            self.reward_active = False
            # 2. 检查 Concession (连续两轮提价 > delta)
            if len(self.bid_history) >= 2:
                last_diff = current_bid - self.bid_history[-1]
                prev_diff = self.bid_history[-1] - self.bid_history[-2]
                if last_diff > self.bid_history[-1]*self.delta_ask and prev_diff > self.bid_history[-1]*self.delta_ask:
                    self.state = 'Concession'
                # 3. 检查 PushBack (让步率过低)
                elif concession_rate < self.lambda_threshold:
                    self.state = 'PushBack'
                else:
                    self.state = 'Baseline'
            else:
                # 数据不足时维持 Baseline
                self.state = 'Baseline'

        return prev_state != self.state

    def respond_to_bid(self, bid_price):
        self.interaction_count += 1
        self.bid_history.append(bid_price)

        # 状态迁移
        self._transition_state(bid_price)

        # 根据当前状态调整 Ask Price
        final_ask = self.current_ask

        if self.state == 'PushBack':
            final_ask += final_ask * self.delta_ask
        elif self.state == 'Concession':
            final_ask -= final_ask * self.gamma
        elif self.state == 'RewardTrust':
            final_ask = self.current_ask * (1 - self.discount_rate)

        # 更新当前 Ask 以供下一轮参考 (平滑更新，避免震荡过大)
        self.current_ask = final_ask
        self.prev_bid = bid_price

        # 生成响应
        if bid_price >= final_ask:
            return {
                "status": "accept",
                "price": bid_price,
                "state": self.state
            }
        else:
            return {
                "status": "counter_offer",
                "counter_offer": final_ask,
                "state": self.state
            }


# ==========================================
# 2. 增强型买方系统 (Simulation 2 - Compatible)
# ==========================================
class BuyerSystemSim2:
    def __init__(self, sellers, total_target=1000, llm_engine=None):
        self.total_target = total_target
        self.procured_units = 0
        self.total_spent = 0
        self.sellers = sellers
        self.llm_engine = llm_engine

        self.beliefs = {s.seller_id: 0.5 for s in sellers}
        self.market_composition_belief = 0.5
        self.interaction_counts = {s.seller_id: 0 for s in sellers}

        # 新增：记录 Seller 状态日志
        self.seller_states_log = {s.seller_id: [] for s in sellers}

    def update_market_belief(self):
        self.market_composition_belief = sum(self.beliefs.values()) / len(self.beliefs)

    def orchestrator_plan(self):
        self.update_market_belief()
        # 策略性放弃：当市场被判定为 Hard 时，过滤低信念卖家
        threshold = 0.4 if self.market_composition_belief < 0.45 else 0.1
        active_plan = [(s_id, score) for s_id, score in self.beliefs.items() if score >= threshold]
        strategy = "Conservative" if self.market_composition_belief < 0.45 else "Aggressive"
        return sorted(active_plan, key=lambda x: x[1], reverse=True), strategy

    def negotiation_agent_action(self, seller_id, belief_score, strategy):
        if self.llm_engine:
            response = self.llm_engine.generate_answer(f"ID:{seller_id}, Mode:{strategy}", belief_score,
                                                       self.procured_units)
            match = re.search(r"FINAL_BID:\s*(\d+)", response)
            if match: return int(match.group(1))
        return 115 if belief_score > 0.6 else 145

    def run_simulation(self, rounds_limit=20, penalty=0.2):
        rounds = 0
        # 记录每轮的 Ask 价格用于收敛判断
        self.round_ask_prices = []

        while self.procured_units < self.total_target and rounds < rounds_limit:
            rounds += 1
            plan, strategy = self.orchestrator_plan()
            round_asks = []

            for seller_id, score in plan:
                if self.procured_units >= self.total_target: break

                self.interaction_counts[seller_id] += 1
                bid = self.negotiation_agent_action(seller_id, score, strategy)
                target_seller = next(s for s in self.sellers if s.seller_id == seller_id)

                # 调用 FSM Seller
                result = target_seller.respond_to_bid(bid)

                # 记录状态
                self.seller_states_log[seller_id].append(result['state'])
                if 'counter_offer' in result:
                    round_asks.append(result['counter_offer'])

                if result["status"] == "accept":
                    self.total_spent += (result["price"] * 50)
                    self.procured_units += 50
                    # 成功交易，提升对该卖家的信念
                    self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * (1 + penalty))
                else:
                    # 拒绝交易，降低信念
                    self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * (1 - penalty))

            if round_asks:
                self.round_ask_prices.append(np.mean(round_asks))
            else:
                self.round_ask_prices.append(0)

        return rounds


# ==========================================
# 3. 实验引擎 (Batch Experiment)
# ==========================================
def run_batch_experiment(num_runs=200, rounds_limit=20, penalty=0.2, lambda_threshold=0.02,  # 从 0.08 降至 0.02。减少对微小让步的过度反应
                 delta_ask=5,  # 从 7 降至 5。PushBack 时的加价幅度减小，避免价格剧烈震荡
                 gamma=4,  # 让步幅度保持不变
                 discount_rate=0.03):
    results = []
    output_dir = "result/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"🚀 开始 {num_runs} 次大规模循环实验 (FSM Active Seller)...")

    for i in range(1, num_runs + 1):
        # --- 随机生成市场逻辑 ---
        m_type = random.choice(["Hard-dominant", "Flexible-dominant"])

        # 记录具体的卖家构成
        if m_type == "Hard-dominant":
            dist = ["hard", "hard", "hard", "hard", "flexible"]
        else:
            dist = ["flexible", "flexible", "flexible", "flexible", "hard"]

        # 随机打乱卖家顺序以增加实验随机性
        random.shuffle(dist)
        # 实例化 ActiveSeller
        sellers = [ActiveSeller(idx + 1, s_type, lambda_threshold, delta_ask, gamma, discount_rate) for idx, s_type in enumerate(dist)]

        buyer = BuyerSystemSim2(sellers, llm_engine=llm)
        rounds_taken = buyer.run_simulation(rounds_limit, penalty)

        aup = buyer.total_spent / buyer.procured_units if buyer.procured_units > 0 else 0

        # 统计在 Hard 市场中对 Flexible 的识别度
        flex_ids = [s.seller_id for s in sellers if s.type == "flexible"]
        flex_interactions = sum(buyer.interaction_counts[fid] for fid in flex_ids)
        total_interactions = sum(buyer.interaction_counts.values())
        pref_ratio = flex_interactions / total_interactions if total_interactions > 0 else 0

        # --- 计算新增实验指标 ---

        # 1. Final Seller State Distribution
        final_states = [s.state for s in sellers]
        state_counts = Counter(final_states)
        state_dist_str = ",".join([f"{k}:{v}" for k, v in sorted(state_counts.items())])

        # 2. Convergence Round (连续3轮 state 不变 且 ask 变化率 < 0.5%)
        convergence_round = rounds_limit + 1
        logs = buyer.seller_states_log
        asks = buyer.round_ask_prices

        # 检查 state 稳定性
        stable_from_round = -1
        for r in range(3, rounds_taken + 1):
            # 检查所有 seller 在过去3轮 (r-3, r-2, r-1) 状态是否一致
            all_stable = True
            for sid in logs:
                if len(logs[sid]) < r: continue
                if not (logs[sid][r - 3] == logs[sid][r - 2] == logs[sid][r - 1]):
                    all_stable = False
                    break
            if all_stable:
                # 检查 ask 价格变化率
                if r >= 4 and asks[r - 2] > 0 and asks[r - 1] > 0:
                    ask_change_rate = abs(asks[r - 1] - asks[r - 2]) / asks[r - 2]
                    if ask_change_rate < 0.005:
                        convergence_round = r
                        break

        # 3. System Cycle Flag (最后5轮中是否存在长度>=3的重复周期)
        system_cycle_flag = False
        if rounds_taken >= 6:  # 至少要有足够轮次检测
            last_5_rounds_states = {sid: logs[sid][-5:] for sid in logs if len(logs[sid]) >= 5}
            for sid, states in last_5_rounds_states.items():
                # 检查是否存在长度为3的周期 (如 A,B,C,A,B,C)
                for period in range(3, 4):  # 仅检查周期为3的情况
                    if len(states) >= 2 * period:
                        if states[-period:] == states[-2 * period:-period]:
                            system_cycle_flag = True
                            break

        # --- 记录数据节点 ---
        results.append({
            "run_id": i,
            "market_type": m_type,
            "market_distribution": ",".join(dist),  # 记录具体的卖家类型序列
            "final_aup": aup,
            "rounds": rounds_taken,
            "final_market_belief": buyer.market_composition_belief,
            "focus_on_flexible_ratio": pref_ratio,
            "final_seller_state_distribution": state_dist_str,
            "convergence_round": convergence_round,
            "system_cycle_flag": system_cycle_flag
        })

        if i % 10 == 0:
            print(f"进度: {i}/{num_runs} | 当前 AUP 均值: {pd.DataFrame(results)['final_aup'].mean():.2f}")
            pd.DataFrame(results).to_csv(
                f"{output_dir}/sim3_fsm_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_intermediate_stats.csv",
                index=False)

    return pd.DataFrame(results)


# ==========================================
# 4. 统计可视化分析
# ==========================================
def analyze_200_runs(df, limit):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 5))

    # 1. AUP 密度分布 (分市场类型)
    plt.subplot(1, 4, 1)
    sns.histplot(data=df, x="final_aup", hue="market_type", kde=True, palette="husl")
    plt.title("AUP Distribution Density")

    # 2. 策略倾向性分析 (Flexible Interaction Ratio)
    plt.subplot(1, 4, 2)
    sns.boxplot(x="market_type", y="focus_on_flexible_ratio", data=df, palette="Pastel1")
    plt.title("Strategic Focus on Flexible Sellers")

    # 3. 学习效果：信念 vs 成本
    plt.subplot(1, 4, 3)
    sns.regplot(data=df[df['market_type'] == "Hard-dominant"], x="final_market_belief", y="final_aup",
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, label="Hard Market Trend")
    plt.title("Learning Efficiency (Belief vs Cost)")
    plt.legend()

    # 4. 新增：收敛轮数分布直方图
    plt.subplot(1, 4, 4)
    # 标记收敛率 (假设默认 rounds_limit 为 15)
    converged_count = len(df[df['convergence_round'] <= limit])
    convergence_rate = converged_count / len(df)

    sns.histplot(data=df, x="convergence_round", hue="market_type", multiple="stack", palette="coolwarm")
    plt.title(f"Convergence Round Distribution\n(Rate: {convergence_rate:.1%})")
    plt.axvline(x=limit, color='black', linestyle='--', label='Limit')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"result/sim3_fsm_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_final_analysis.png")
    plt.show()


if __name__ == "__main__":
    # 运行 200 次
    llm = GPTInference()
    num_runs = 200
    rounds_limit = 20
    penalty = 0.3
    lambda_threshold = 0.01  # [优化] 从 0.08 降至 0.02。减少对微小让步的过度反应
    delta_ask = 2  # [优化] 从 7 降至 5。PushBack 时的加价幅度减小，避免价格剧烈震荡
    # df_final = run_batch_experiment(num_runs, rounds_limit, penalty, lambda_threshold, delta_ask)
    #
    # # 保存最终详细结果
    # df_final.to_csv(f"result/sim3_fsm_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_full_data.csv",
    #                 index=False)
    df_final = read_csv(
        f"result/sim3_fsm_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_full_data.csv")  # 重新读取以确保数据完整性
    # 分析
    analyze_200_runs(df_final, rounds_limit)

    # 打印按市场类型分类的描述统计
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY BY MARKET TYPE (FSM SELLER)")
    print("=" * 50)
    print(df_final.groupby('market_type')[['final_aup', 'rounds', 'focus_on_flexible_ratio']].mean())
    print("\nState Distribution Samples:")
    print(df_final['final_seller_state_distribution'].value_counts().head())