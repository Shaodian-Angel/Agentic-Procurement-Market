import random
import re
import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

matplotlib.use('TkAgg')
import seaborn as sns
from utils.gpt_inference import GPTInference


# ==========================================
# 1. 卖方系统 (Passive Sellers)
# ==========================================
class PassiveSeller:
    def __init__(self, seller_id, seller_type):
        self.seller_id = seller_id
        self.type = seller_type  # "hard" or "flexible"
        self.reservation_price = 150 if seller_type == "hard" else 110

    def respond_to_bid(self, bid_price):
        if bid_price >= self.reservation_price:
            return {"status": "accept", "price": bid_price}
        return {"status": "reject", "counter_offer": self.reservation_price + 5}


# ==========================================
# 2. 增强型买方系统 (Simulation 2)
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
        while self.procured_units < self.total_target and rounds < rounds_limit:
            rounds += 1
            plan, strategy = self.orchestrator_plan()
            for seller_id, score in plan:
                if self.procured_units >= self.total_target: break

                self.interaction_counts[seller_id] += 1
                bid = self.negotiation_agent_action(seller_id, score, strategy)
                target_seller = next(s for s in self.sellers if s.seller_id == seller_id)
                result = target_seller.respond_to_bid(bid)

                if result["status"] == "accept":
                    self.total_spent += (result["price"] * 50)
                    self.procured_units += 50
                    self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * (1+penalty))
                else:
                    self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * (1-penalty))
        return rounds


# ==========================================
# 3. 实验引擎 (记录市场类型)
# ==========================================
def run_batch_experiment(num_runs=200, rounds_limit=20, penalty=0.2):

    results = []
    output_dir = "result/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"🚀 开始 {num_runs} 次大规模循环实验...")

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
        sellers = [PassiveSeller(idx + 1, s_type) for idx, s_type in enumerate(dist)]

        buyer = BuyerSystemSim2(sellers, llm_engine=llm)
        rounds_taken = buyer.run_simulation(rounds_limit, penalty)

        aup = buyer.total_spent / buyer.procured_units if buyer.procured_units > 0 else 0

        # 统计在 Hard 市场中对 Flexible 的识别度
        flex_ids = [s.seller_id for s in sellers if s.type == "flexible"]
        flex_interactions = sum(buyer.interaction_counts[fid] for fid in flex_ids)
        total_interactions = sum(buyer.interaction_counts.values())
        pref_ratio = flex_interactions / total_interactions if total_interactions > 0 else 0

        # --- 记录数据节点 ---
        results.append({
            "run_id": i,
            "market_type": m_type,
            "market_distribution": ",".join(dist),  # 记录具体的卖家类型序列
            "final_aup": aup,
            "rounds": rounds_taken,
            "final_market_belief": buyer.market_composition_belief,
            "focus_on_flexible_ratio": pref_ratio
        })

        if i % 10 == 0:
            print(f"进度: {i}/{num_runs} | 当前 AUP 均值: {pd.DataFrame(results)['final_aup'].mean():.2f}")
            pd.DataFrame(results).to_csv(f"{output_dir}/sim2_intermediate_stats.csv", index=False)

    return pd.DataFrame(results)


# ==========================================
# 4. 统计可视化分析
# ==========================================
def analyze_200_runs(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 5))

    # 1. AUP 密度分布 (分市场类型)
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x="final_aup", hue="market_type", kde=True, palette="husl")
    plt.title("AUP Distribution Density")

    # 2. 策略倾向性分析 (Flexible Interaction Ratio)
    # 这个图能证明系统是否“学会”把精力花在 Flexible 卖家身上
    plt.subplot(1, 3, 2)
    sns.boxplot(x="market_type", y="focus_on_flexible_ratio", data=df, palette="Pastel1")
    plt.title("Strategic Focus on Flexible Sellers")

    # 3. 学习效果：信念 vs 成本
    plt.subplot(1, 3, 3)
    sns.regplot(data=df[df['market_type'] == "Hard-dominant"], x="final_market_belief", y="final_aup",
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, label="Hard Market Trend")
    plt.title("Learning Efficiency (Belief vs Cost)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"result/sim2_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_analysis.png")
    plt.show()


if __name__ == "__main__":
    # 运行 200 次
    llm = GPTInference()
    num_runs = 200
    rounds_limit = 20
    penalty = 0.3
    df_final = run_batch_experiment(num_runs, rounds_limit, penalty)

    # 保存最终详细结果
    df_final.to_csv(f"result/sim2_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_full_data.csv", index=False)
    df_final = read_csv(f"result/sim2_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_full_data.csv")  # 重新读取以确保数据完整性
    # 分析
    analyze_200_runs(df_final)

    # 打印按市场类型分类的描述统计
    print("\n" + "=" * 40)
    print("STATISTICAL SUMMARY BY MARKET TYPE")
    print("=" * 40)
    print(df_final.groupby('market_type')[['final_aup', 'rounds', 'focus_on_flexible_ratio']].mean())