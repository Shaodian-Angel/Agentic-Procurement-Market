import random
import re
import os
import time
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv

# 确保你的项目根目录下有这两个模块
from logger import logger
from utils.gpt_inference import GPTInference


# ==========================================
# 1. 动态自适应卖方系统 (Active Sellers with Capacity Constraints)
# ==========================================
class DynamicConstrainedSeller:
    def __init__(self, seller_id, seller_type, initial_capacity=400):
        self.seller_id = seller_id
        self.type = seller_type  # "hard" or "flexible"
        self.reservation_price = 150 if seller_type == "hard" else 110
        self.initial_ask = 180

        # 产能约束与稀缺度设定
        self.capacity = initial_capacity
        self.initial_capacity = initial_capacity

    def respond_to_bid(self, bid_price, requested_units):
        """核心逻辑：卖方根据剩余产能的稀缺度，动态调整心理底价"""
        if self.capacity <= 0:
            return {"status": "reject", "counter_offer": 999, "available": 0}

        actual_units = min(requested_units, self.capacity)

        # 稀缺度溢价：产能越少，底价越高 (最高上浮 15%)
        scarcity_factor = 1.0 + (1.0 - (self.capacity / self.initial_capacity)) * 0.15
        dynamic_reservation = self.reservation_price * scarcity_factor

        if bid_price >= dynamic_reservation:
            self.capacity -= actual_units
            return {
                "status": "accept",
                "price": bid_price,
                "units": actual_units
            }
        else:
            counter = max(dynamic_reservation, bid_price + 10)
            return {
                "status": "reject",
                "counter_offer": round(counter, 2),
                "available": actual_units
            }


# ==========================================
# 2. 宏观协同买方系统 (Macro-Coordinator Buyer)
# ==========================================
class CoordinatorBuyerSystem:
    def __init__(self, sellers, total_target=1000, llm_engine=None):
        self.total_target = total_target
        self.procured_units = 0
        self.total_spent = 0
        self.sellers = sellers
        self.llm_engine = llm_engine

        self.beliefs = {s.seller_id: 0.5 for s in sellers}
        self.market_composition_belief = 0.5

        self.interaction_counts = {s.seller_id: 0 for s in sellers}
        self.quota_allocations = {s.seller_id: [] for s in sellers}

        # 新增：用于记录每轮详细谈判对话的列表
        self.detail_dialog = []

    def update_market_belief(self):
        self.market_composition_belief = sum(self.beliefs.values()) / len(self.beliefs)

    def macro_coordination_plan(self):
        """Simulation 4 核心：跨卖家产能配额分配 (Quota Allocation)"""
        self.update_market_belief()
        threshold = 0.4 if self.market_composition_belief < 0.45 else 0.1
        active_sellers = [s for s in self.sellers if self.beliefs[s.seller_id] >= threshold and s.capacity > 0]

        if not active_sellers:
            active_sellers = [
                max((s for s in self.sellers if s.capacity > 0), key=lambda s: self.beliefs[s.seller_id], default=None)]
            if active_sellers[0] is None: return []  # 市场彻底没货

        # 动态权重：信任度 * 产能充裕度
        total_weight = 0
        seller_weights = {}
        for s in active_sellers:
            weight = self.beliefs[s.seller_id] * (s.capacity / s.initial_capacity + 0.1)
            seller_weights[s.seller_id] = weight
            total_weight += weight

        remaining_target = self.total_target - self.procured_units
        plan = []
        for s_id, w in seller_weights.items():
            allocated_quota = min(80, round((w / total_weight) * remaining_target))
            allocated_quota = max(10, allocated_quota)
            plan.append((s_id, allocated_quota, self.beliefs[s_id]))

        return sorted(plan, key=lambda x: (x[2], x[1]), reverse=True)

    def negotiation_agent_action(self, seller_id, belief_score, allocated_quota):
        if self.llm_engine:
            prompt_context = f"ID:{seller_id}, TargetQuota:{allocated_quota}"
            response = self.llm_engine.generate_answer(prompt_context, belief_score, self.procured_units)
            match = re.search(r"FINAL_BID:\s*(\d+)", response)
            if match: return int(match.group(1))

        if allocated_quota > 50 and belief_score > 0.6:
            return 125
        return 115 if belief_score > 0.6 else 145

    def run_simulation(self):
        rounds = 0
        while self.procured_units < self.total_target and rounds < 25:
            rounds += 1
            plan = self.macro_coordination_plan()
            if not plan: break

            for seller_id, quota, score in plan:
                if self.procured_units >= self.total_target: break

                target_seller = next(s for s in self.sellers if s.seller_id == seller_id)
                self.interaction_counts[seller_id] += 1
                self.quota_allocations[seller_id].append(quota)

                bid = self.negotiation_agent_action(seller_id, score, quota)
                result = target_seller.respond_to_bid(bid, quota)

                # --- 细颗粒度对话历史文本构建 ---
                dialog_node = f"[Round {rounds}] Buyer targets Seller {seller_id} ({target_seller.type}) with Allocation Quota: {quota}. " \
                              f"Buyer Bids: ${bid}."

                if result["status"] == "accept":
                    self.total_spent += (result["price"] * result["units"])
                    self.procured_units += result["units"]
                    self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * 1.2)

                    dialog_node += f" -> RESPONSE: ACCEPT. Successfully procured {result['units']} units @ ${result['price']}."
                else:
                    self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * 0.75)

                    dialog_node += f" -> RESPONSE: REJECT. Seller countered with ${result['counter_offer']}."

                self.detail_dialog.append(dialog_node)

        return rounds


# ==========================================
# 3. 实验引擎 (多目标与对话记录)
# ==========================================
def run_batch_experiment_sim4(num_runs=200):
    llm = GPTInference()
    results = []
    output_dir = "result/sim4"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"🚀 开始 Simulation 4: {num_runs} 次产能约束多目标实验(含细粒度对话追踪)...")

    for i in range(1, num_runs + 1):
        market_type = random.choice(["Scarcity-Hard", "Abundant-Flex"])

        if market_type == "Scarcity-Hard":
            dist = [("hard", 250), ("hard", 250), ("hard", 250), ("hard", 250), ("flexible", 500)]
        else:
            dist = [("flexible", 350), ("flexible", 350), ("flexible", 350), ("flexible", 350), ("hard", 200)]

        random.shuffle(dist)
        sellers = [DynamicConstrainedSeller(idx + 1, s_type, cap) for idx, (s_type, cap) in enumerate(dist)]

        buyer = CoordinatorBuyerSystem(sellers, llm_engine=llm)
        rounds_taken = buyer.run_simulation()

        aup = buyer.total_spent / buyer.procured_units if buyer.procured_units > 0 else 0

        # --- 数据节点：合并存储全量对话轨迹 ---
        results.append({
            "run_id": i,
            "market_type": market_type,
            "final_aup": aup,
            "time_steps_needed": rounds_taken,
            "procured_units": buyer.procured_units,
            "final_market_belief": buyer.market_composition_belief,
            "detail_dialog": " || ".join(buyer.detail_dialog)  # 用双竖线清晰切分每轮会话
        })

        if i % 10 == 0:
            print(f"进度: {i}/{num_runs} | 均价 AUP: {pd.DataFrame(results)['final_aup'].mean():.2f}")
            pd.DataFrame(results).to_csv(f"{output_dir}/sim4_intermediate_stats.csv", index=False)

    return pd.DataFrame(results)


# ==========================================
# 4. 多目标可视化分析
# ==========================================
def analyze_sim4_runs(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 最终成本分布 (AUP)
    sns.boxplot(ax=axes[0], x='market_type', y='final_aup', data=df, palette="Oranges")
    axes[0].set_title("Cost Optimization (Final AUP)")
    axes[0].set_ylabel("Average Unit Price ($)")

    # 2. 交付效率分布 (Rounds Taken)
    sns.countplot(ax=axes[1], x='time_steps_needed', hue='market_type', data=df, palette="Blues")
    axes[1].set_title("Supply Chain Velocity")
    axes[1].set_xlabel("Rounds to Procure Target")

    # 3. 帕累托前沿：成本与时间的博弈
    sns.scatterplot(ax=axes[2], data=df, x="time_steps_needed", y="final_aup", hue="market_type", alpha=0.7,
                    palette="Set1")
    axes[2].set_title("Multi-Objective Trade-off (Cost vs. Velocity)")
    axes[2].set_xlabel("Procurement Speed (Rounds)")
    axes[2].set_ylabel("Procurement Cost (AUP)")

    plt.tight_layout()
    plt.savefig("result/sim4/sim4_final_200_runs_analysis.png")
    plt.show()


if __name__ == "__main__":
    df_final = run_batch_experiment_sim4(200)
    df_final.to_csv("result/sim4/sim4_final_200_runs_full_data.csv", index=False)

    df_final = read_csv("result/sim4/sim4_final_200_runs_full_data.csv")
    analyze_sim4_runs(df_final)

    print("\n" + "=" * 40)
    print("SIMULATION 4 PERFORMANCE SUMMARY")
    print("=" * 40)
    print(df_final.groupby('market_type')[['final_aup', 'time_steps_needed', 'procured_units']].mean())