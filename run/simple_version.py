import ast
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
        """
        卖方根据剩余产能的稀缺度，动态调整心理底价。
        为了配合步骤5的LLM信号提取，卖方现在返回自然语言文本。
        """
        if self.capacity <= 0:
            return "REJECT: Out of stock. Capacity completely exhausted."

        actual_units = min(requested_units, self.capacity)

        # 稀缺度溢价：产能越少，底价越高 (最高上浮 15%)
        scarcity_factor = 1.0 + (1.0 - (self.capacity / self.initial_capacity)) * 0.15
        dynamic_reservation = self.reservation_price * scarcity_factor

        if bid_price >= dynamic_reservation:
            return f"ACCEPT: Allocation of {actual_units} units granted at ${bid_price:.2f} per unit."
        else:
            counter = max(dynamic_reservation, bid_price + 10)
            return f"COUNTER_OFFER: Proposal of ${counter:.2f} per unit for {actual_units} units."


# ==========================================
# 2. 宏观协同买方系统 (8-Step Orchestrator Version)
# ==========================================
class CoordinatorBuyerSystem:
    def __init__(self, sellers, total_target=1000, llm_engine=None):
        self.total_target = total_target
        self.procured_units = 0
        self.total_spent = 0
        self.sellers = sellers
        self.llm_engine = llm_engine

        # --- MAINTAINS 状态维护区 ---
        # 1. 针对每个卖家维护贝叶斯先验参数（Alpha: 表现出Hard的计数, Beta: 表现出Flexible的计数）
        self.alphas = {s.seller_id: 1.0 for s in sellers}
        self.betas = {s.seller_id: 1.0 for s in sellers}
        self.beliefs = {s.seller_id: 0.5 for s in sellers}  # P(hard-type)

        # 2. 谈判历史与最佳报价记录
        self.offer_history = {s.seller_id: [] for s in sellers}
        self.best_offers_seen = {s.seller_id: float('inf') for s in sellers}

        # 3. 截止日期前剩余轮数
        self.rounds_remaining = 25

        # 4. 迄今组装的最优可行捆绑包
        self.best_feasible_bundle = []

        # 基础统计辅助变量
        self.interaction_counts = {s.seller_id: 0 for s in sellers}
        self.quota_allocations = {s.seller_id: [] for s in sellers}
        self.detail_dialog = []

    def update_beliefs(self):
        """计算明面上的 P(hard-type) 期望值并更新市场整体宏观信念"""
        for s_id in self.alphas:
            # Beta分布中属于 Hard 类型的概率期望公式
            self.beliefs[s_id] = self.alphas[s_id] / (self.alphas[s_id] + self.betas[s_id])

        # 动态计算整个市场的平均硬性倾向，为后续实验提供准确的宏观数据节点
        self.market_composition_belief = sum(self.beliefs.values()) / len(self.beliefs)

    def run_simulation(self):
        while self.procured_units < self.total_target and self.rounds_remaining > 0:
            current_round = 26 - self.rounds_remaining
            self.update_beliefs()

            # ----------------------------------------------------
            # 1. SELECT: 宏微观联动的自适应汤普森采样
            # ----------------------------------------------------
            sampled_hard_probabilities = {}
            selected_sellers = []

            # 【核心联动】根据宏观市场信念，动态调整对硬派（Hard）卖家的容忍度阈值
            # 如果大盘整体很硬 (>= 0.55)，容忍度放宽到 0.90（死马当活马医）
            # 如果大盘整体很软 (< 0.55)，容忍度收紧到 0.70（精挑细选）
            adaptive_threshold = 0.90 if self.market_composition_belief >= 0.55 else 0.70

            for s in self.sellers:
                if s.capacity > 0:
                    # 微观层面：从 Beta 分布中对该卖家 P(hard) 进行随机采样
                    sampled_p = np.random.beta(self.alphas[s.seller_id], self.betas[s.seller_id])
                    sampled_hard_probabilities[s.seller_id] = sampled_p

                    # 结合自适应阈值进行筛选, 汤普森选择策略：倾向于避开采样到高刚性(Hard)的卖家，设置阈值筛选
                    if sampled_p <= adaptive_threshold:
                        selected_sellers.append(s)

            if not selected_sellers:  # 兜底：如果全部被筛掉，选产能最大的一个
                selected_sellers = [
                    max((s for s in self.sellers if s.capacity > 0), key=lambda s: s.capacity, default=None)]
                if selected_sellers[0] is None: break

            # 计算本轮分配基础配额
            remaining_needed = self.total_target - self.procured_units
            base_quota = max(10, min(80, round(remaining_needed / len(selected_sellers))))

            # ----------------------------------------------------
            # 2. CHOOSE: 依据 P(hard-type) 为每个选定卖家匹配策略
            # ----------------------------------------------------
            strategies = {}
            for s in selected_sellers:
                p_hard = self.beliefs[s.seller_id]
                if p_hard > 0.7:
                    strategies[s.seller_id] = "H-strategy"  # 对方太硬，高出价求稳
                elif p_hard < 0.3:
                    strategies[s.seller_id] = "F-strategy"  # 对方很软，极限压价
                else:
                    strategies[s.seller_id] = "probe"  # 态度模糊，中等价位试探

            # ----------------------------------------------------
            # 3. SEND & 4. RECEIVE: 发送Offer并同步限时收集响应
            # ----------------------------------------------------
            raw_responses = {}
            bids_sent = {}
            for s in selected_sellers:
                s_id = s.seller_id
                self.interaction_counts[s_id] += 1
                self.quota_allocations[s_id].append(base_quota)

                # 依据策略确定买方出价
                if self.llm_engine:
                    prompt_context = f"ID:{s_id}, TargetQuota:{base_quota}, Strategy:{strategies[s_id]}"
                    response_text = self.llm_engine.generate_answer(prompt_context, self.beliefs[s_id],
                                                                    self.procured_units)
                    match = re.search(r"FINAL_BID:\s*(\d+)", response_text)
                    bid_price = int(match.group(1)) if match else (155 if strategies[s_id] == "H-strategy" else 115)
                else:
                    if strategies[s_id] == "H-strategy":
                        bid_price = 155
                    elif strategies[s_id] == "F-strategy":
                        bid_price = 115
                    else:
                        bid_price = 135

                bids_sent[s_id] = bid_price
                # 基础设施模拟：调用接口并带超时（此处为同步流式模拟接收）
                raw_responses[s_id] = s.respond_to_bid(bid_price, base_quota)

            # ----------------------------------------------------
            # 5. INTERPRET: 解析提炼非结构化响应信号
            # ----------------------------------------------------
            structured_signals = {}
            for s_id, text_resp in raw_responses.items():
                # 提取数字与状态
                if "ACCEPT" in text_resp:
                    units = int(re.search(r"of (\d+) units", text_resp).group(1))
                    price = float(re.search(r"\$(\d+\.\d+)", text_resp).group(1))
                    structured_signals[s_id] = {"status": "accept", "price": price, "units": units}
                elif "COUNTER_OFFER" in text_resp:
                    price = float(re.search(r"\$(\d+\.\d+)", text_resp).group(1))
                    units = int(re.search(r"for (\d+) units", text_resp).group(1))
                    structured_signals[s_id] = {"status": "counter", "price": price, "units": units}
                else:
                    structured_signals[s_id] = {"status": "reject", "price": 0, "units": 0}

                # 更新维护记录
                self.offer_history[s_id].append(structured_signals[s_id])
                if structured_signals[s_id]["price"] > 0:
                    self.best_offers_seen[s_id] = min(self.best_offers_seen[s_id], structured_signals[s_id]["price"])

            # ----------------------------------------------------
            # 6. UPDATE: 基于响应结果执行贝叶斯更新
            # ----------------------------------------------------
            for s_id, signal in structured_signals.items():
                if signal["status"] == "accept":
                    # 卖家接受了出价，说明其不属于刚性硬派，更新 Flexible 分布参数
                    self.betas[s_id] += 1.0
                else:
                    # 拒绝或开出高反向报价，增加 Hard 分布参数计数
                    self.alphas[s_id] += 1.0

            # ----------------------------------------------------
            # 7. ASSEMBLE: 约束优化，贪婪法组装当前最优可行捆绑包
            # ----------------------------------------------------
            # 筛选当前有效的成功报价进行贪婪打包
            valid_round_offers = [
                {"seller_id": s_id, "price": sig["price"], "units": sig["units"]}
                for s_id, sig in structured_signals.items() if sig["status"] == "accept"
            ]
            # 价格从低到高排序
            valid_round_offers.sort(key=lambda x: x["price"])

            executed_units_this_round = 0
            for offer in valid_round_offers:
                needed = self.total_target - self.procured_units
                if needed <= 0: break

                take_units = min(offer["units"], needed)
                if take_units > 0:
                    # 实际物理扣减卖方产能并最终计入捆绑包
                    target_seller = next(s for s in self.sellers if s.seller_id == offer["seller_id"])
                    target_seller.capacity -= take_units

                    self.total_spent += offer["price"] * take_units
                    self.procured_units += take_units
                    executed_units_this_round += take_units

                    self.best_feasible_bundle.append(offer)

                # 对话日志结构化同步
                dialog_node = f"[Round {current_round}] Seller_ID: {offer['seller_id']}. Strategy: {strategies[offer['seller_id']]}. " \
                              f"Bids ${bids_sent[offer['seller_id']]} -> ACCEPT. Got {take_units} units."
                self.detail_dialog.append(dialog_node)

            # 补充未成交的日志追踪
            for s_id, sig in structured_signals.items():
                if sig["status"] != "accept":
                    dialog_node = f"[Round {current_round}] Seller_ID: {s_id}. Strategy: {strategies[s_id]}. " \
                                  f"Bids ${bids_sent[s_id]} -> REJECT/COUNTER."
                    self.detail_dialog.append(dialog_node)

            # ----------------------------------------------------
            # 8. DECIDE: 终止判定 (检查目标或终止性价比)
            # ----------------------------------------------------
            self.rounds_remaining -= 1
            if self.procured_units >= self.total_target:
                break  # 目标达成，提前主动终止
            if self.rounds_remaining <= 0:
                break  # 达到截止日期限制，强行退出

        return 25 - self.rounds_remaining


# ==========================================
# 3. 实验引擎 (保持原样)
# ==========================================
def run_batch_experiment_simple(num_runs=200):
    llm = GPTInference()
    results = []
    output_dir = "result/simple"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"🚀 开始 Simulation: {num_runs} 次产能约束多目标实验(含细粒度对话追踪)...")

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

        results.append({
            "run_id": i,
            "market_type": market_type,
            "final_aup": aup,
            "time_steps_needed": rounds_taken,
            "procured_units": buyer.procured_units,
            "final_market_belief": buyer.beliefs,
            "detail_dialog": " || ".join(buyer.detail_dialog)
        })

        if i % 10 == 0:
            print(f"进度: {i}/{num_runs} | 均价 AUP: {pd.DataFrame(results)['final_aup'].mean():.2f}")
            pd.DataFrame(results).to_csv(f"{output_dir}/sim_intermediate_stats.csv", index=False)

    return pd.DataFrame(results)


# ==========================================
# 4. 多目标可视化分析 (保持原样)
# ==========================================
def analyze_simple_runs(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(ax=axes[0], x='market_type', y='final_aup', data=df, palette="Oranges")
    axes[0].set_title("Cost Optimization (Final AUP)")
    axes[0].set_ylabel("Average Unit Price ($)")

    sns.countplot(ax=axes[1], x='time_steps_needed', hue='market_type', data=df, palette="Blues")
    axes[1].set_title("Supply Chain Velocity")
    axes[1].set_xlabel("Rounds to Procure Target")

    sns.scatterplot(ax=axes[2], data=df, x="time_steps_needed", y="final_aup", hue="market_type", alpha=0.7,
                    palette="Set1")
    axes[2].set_title("Multi-Objective Trade-off (Cost vs. Velocity)")
    axes[2].set_xlabel("Procurement Speed (Rounds)")
    axes[2].set_ylabel("Procurement Cost (AUP)")

    plt.tight_layout()
    plt.savefig("result/simple/sim_final_200_runs_analysis.png")
    plt.show()


# Parse beliefs
def extract_beliefs(df):
    try:
        belief_dict = ast.literal_eval(df)
        return list(belief_dict.values())
    except Exception:
        return []


def show_equilibrium_analysis(df):

    df['belief_list'] = df['final_market_belief'].apply(extract_beliefs)

    # Explode beliefs for the third plot
    df_beliefs = df.explode('belief_list')
    df_beliefs['belief_list'] = df_beliefs['belief_list'].astype(float)

    # Set style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Price Equilibrium Distribution
    sns.violinplot(data=df, x='market_type', y='final_aup', ax=axes[0], palette="Set2", inner="box")
    axes[0].set_title("1. Price Equilibrium Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Market Regime", fontsize=12)
    axes[0].set_ylabel("Final Average Unit Price (AUP)", fontsize=12)

    # Plot 2: Convergence Path (Time vs Price)
    sns.scatterplot(data=df, x='time_steps_needed', y='final_aup', hue='market_type', ax=axes[1], palette="Set2", s=70,
                    alpha=0.8)
    axes[1].set_title("2. Negotiation Convergence Speed vs. Price", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Time Steps to Convergence", fontsize=12)
    axes[1].set_ylabel("Final Average Unit Price (AUP)", fontsize=12)
    axes[1].legend(title="Market Type")

    # Plot 3: Belief Polarization/Convergence
    sns.kdeplot(data=df_beliefs, x='belief_list', hue='market_type', ax=axes[2], palette="Set2", fill=True,
                common_norm=False, alpha=0.5)
    axes[2].set_title("3. Final Agent Market Belief Distribution", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Market Belief State", fontsize=12)
    axes[2].set_ylabel("Density", fontsize=12)

    plt.tight_layout()
    plt.savefig("result/simple/equilibrium_analysis.png", dpi=300)
    print("Plot successfully saved as equilibrium_analysis.png")
    plt.show()


if __name__ == "__main__":
    # df_final = run_batch_experiment_simple(200)
    # df_final.to_csv("result/simple/sim_final_200_runs_full_data.csv", index=False)

    df_final = read_csv("result/simple/sim_final_200_runs_full_data.csv")
    analyze_simple_runs(df_final)
    show_equilibrium_analysis(df_final)

    print("\n" + "=" * 40)
    print("SIMULATION PERFORMANCE SUMMARY")
    print("=" * 40)
    print(df_final.groupby('market_type')[['final_aup', 'time_steps_needed', 'procured_units']].mean())