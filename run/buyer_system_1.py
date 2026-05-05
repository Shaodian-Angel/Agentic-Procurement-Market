import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger import logger
from pandas import read_csv

from utils.gpt_inference import GPTInference


def call_your_llm(seller_id, belief_score, procured_units) -> str:
    """
    在此处接入您的 LLM API (如 OpenAI, Anthropic, 或本地模型)。
    该函数应接收提示词并返回模型的文本响应。
    """
    response = llm_inference.generate_answer(seller_id, belief_score, procured_units)
    # 示例: response = openai.ChatCompletion.create(...)
    return response


# ==========================================
# 1. 卖方系统 (Passive Sellers)
# ==========================================
class PassiveSeller:
    def __init__(self, seller_id, seller_type):
        self.seller_id = seller_id
        self.type = seller_type  # "hard" 或 "flexible"
        # 设定底价：Hard 卖家高，Flexible 卖家低
        self.reservation_price = 150 if seller_type == "hard" else 110
        self.initial_ask = 180

    def respond_to_bid(self, bid_price):
        """
        核心方法：判断是否接受出价
        """
        # 如果出价高于或等于底价，则成交
        if bid_price >= self.reservation_price:
            return {
                "status": "accept",
                "price": bid_price  # 返回实际成交价，即买方的出价
            }
        else:
            # 否则拒绝，并给出一个反向报价 (Counter-offer)
            return {
                "status": "reject",
                "counter_offer": self.reservation_price + 5
            }


# ==========================================
# 2. 买方 Multi-Agent 系统
# ==========================================
class BuyerSystem:
    def __init__(self, sellers, total_target=1000):
        self.total_target = total_target
        self.procured_units = 0
        self.total_spent = 0  # 用于累加总支出
        self.sellers = sellers
        self.beliefs = {s.seller_id: 0.5 for s in sellers}
        self.transaction_log = []  # 记录每笔成交详情

    def orchestrator_plan(self):
        # 优先分配给信念高的卖家
        sorted_sellers = sorted(self.beliefs.items(), key=lambda x: x[1], reverse=True)
        return sorted_sellers

    def negotiation_agent_action(self,seller_id, belief_score):
        """
        此处为接入 LLM 的位置
        """
        # 调用LLM API
        response = call_your_llm(seller_id, belief_score, self.procured_units)

        # 解析逻辑
        try:
            import re
            match = re.search(r"FINAL_BID:\s*(\d+)", response)
            if match:
                bid = int(match.group(1))
                return bid
        except:
            pass
        return 130  # 默认保底价

    def update_belief(self, seller_id, outcome):
        if outcome["status"] == "accept":
            self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * 1.2)
        else:
            self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * 0.8)

    def run_step(self, run_id, round_num):
        """
        执行一个采样/谈判周期，并精确记录成本。
        增加输出节点的运行函数
        """
        plan = self.orchestrator_plan()

        # 节点 1: Orchestrator 计划输出
        top_seller = plan[0][0]
        top_score = plan[0][1]
        logger.info(f"  [Round {round_num}] Orchestrator 倾向于 Seller {top_seller} (信心值: {top_score:.2f})")

        for seller_id, score in plan:
            if self.procured_units >= self.total_target:
                break

            bid = self.negotiation_agent_action(seller_id, score)
            target_seller = next(s for s in self.sellers if s.seller_id == seller_id)
            result = target_seller.respond_to_bid(bid)

            # 节点 2: 谈判明细输出
            status_icon = "✅" if result["status"] == "accept" else "❌"
            logger.info(
                f"    - Negotiating with Seller {seller_id} ({target_seller.type}): Bid ${bid} -> {status_icon} {result['status']}")

            self.update_belief(seller_id, result)

            if result["status"] == "accept":
                deal_price = result["price"]
                deal_units = 50
                self.total_spent += (deal_price * deal_units)
                self.procured_units += deal_units

                self.transaction_log.append({
                    "run_id": run_id,
                    "seller_id": seller_id,
                    "price": deal_price,
                    "units": deal_units,
                    "seller_type": target_seller.type
                })


# ==========================================
# 3. 实验环境：增加进度监控
# ==========================================
def execute_baseline_study(num_runs=30):
    baseline_stats = []

    logger.info(f"\n🚀 开始 Baseline 实验: 共 {num_runs} 次独立运行")
    logger.info("=" * 50)

    for run_i in range(num_runs):
        market = [
            PassiveSeller(1, "hard"), PassiveSeller(2, "flexible"),
            PassiveSeller(3, "hard"), PassiveSeller(4, "flexible"),
            PassiveSeller(5, "flexible")
        ]
        buyer = BuyerSystem(market)

        logger.info(f"\n▶️ Run {run_i + 1}/{num_runs} 启动...")

        rounds = 0
        while buyer.procured_units < 1000 and rounds < 20:
            rounds += 1
            buyer.run_step(run_i + 1, rounds)

        # 节点 3: 单次实验结果总结
        aup = buyer.total_spent / buyer.procured_units if buyer.procured_units > 0 else 0
        logger.info(
            f"🚩 Run {run_i + 1} 结束 | 进度: {buyer.procured_units}/1000 | 平均成本 (AUP): ${aup:.2f} | 耗时: {rounds} 轮")

        baseline_stats.append({
            "run_id": run_i + 1,
            "avg_unit_price": aup,
            "total_cost": buyer.total_spent,
            "rounds_to_complete": rounds
        })

    logger.info("\n" + "=" * 50)
    logger.info("✅ 所有实验运行完毕，准备生成分布图...")
    return pd.DataFrame(baseline_stats)


# ==========================================
# 4. 结果可视化：Cost Distribution
# ==========================================
def plot_baseline_distribution(df):
    plt.figure(figsize=(10, 6))

    # 绘制成本分布直方图和核密度估计
    sns.histplot(df['avg_unit_price'], kde=True, color="royalblue", bins=10)

    # 标注均值
    mean_val = df['avg_unit_price'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean AUP: {mean_val:.2f}')

    plt.title("Simulation 1: Baseline Cost Distribution", fontsize=14)
    plt.xlabel("Average Unit Price (AUP)")
    plt.ylabel("Frequency (Number of Runs)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("result/sim1_final_200_runs_analysis.png")
    plt.show()


# ==========================================
# 5. 主程序执行
# ==========================================
if __name__ == "__main__":
    llm_inference = GPTInference()
    # 1. 运行实验
    df_baseline = execute_baseline_study(200)
    # 2. 保存结果为基准文件
    df_baseline.to_csv(f"result/sim1_baseline_results_{time.time()}.csv", index=False)
    df_baseline = read_csv("result/sim1_baseline_results.csv")  # 替换为实际文件名
    # 3. 分析与可视化
    plot_baseline_distribution(df_baseline)