import random
from utils.gpt_inference import GPTInference


# ==========================================
# 1. 大语言模型接入点 (LLM Placeholder)
# ==========================================

def call_your_llm(seller_id, belief_score, procured_units) -> str:
    """
    在此处接入您的 LLM API (如 OpenAI, Anthropic, 或本地模型)。
    该函数应接收提示词并返回模型的文本响应。
    """
    response = llm_inference.generate_answer(seller_id, belief_score, procured_units)
    # 示例: response = openai.ChatCompletion.create(...)
    return response


# ==========================================
# 2. 卖方系统 (Passive Sellers)
# ==========================================
class PassiveSeller:
    def __init__(self, seller_id, seller_type):
        self.seller_id = seller_id
        self.type = seller_type  # "hard" 或 "flexible"
        # 设置固定底价
        self.reservation_price = 150 if seller_type == "hard" else 110
        self.initial_ask = 180

    def respond_to_bid(self, bid_price):
        """
        Simulation 1 设定：卖方是非智能的，仅根据底价和简单逻辑回应 。
        """
        if bid_price >= self.reservation_price:
            return {"status": "accept", "price": bid_price}
        else:
            # 简单回绝或坚持原价
            return {"status": "reject", "counter_offer": self.reservation_price + 10}


# ==========================================
# 3. 买方 Multi-Agent 系统
# ==========================================
class BuyerSystem:
    def __init__(self, sellers, total_target=1000):
        self.total_target = total_target  # 采购目标 1,000 单位
        self.procured_units = 0
        self.sellers = sellers
        # 初始信念：对每个 seller 是 flexible 的概率设为 0.5
        self.beliefs = {s.seller_id: 0.5 for s in sellers}

    def orchestrator_plan(self):
        """
        Orchestrator 职能：决定将采购精力分配给哪些潜力卖方 [cite: 6, 10]。
        """
        # 简单策略：优先分配给信念中更可能是 "flexible" 的卖家
        sorted_sellers = sorted(self.beliefs.items(), key=lambda x: x[1], reverse=True)
        return sorted_sellers

    def negotiation_agent_action(self, seller_id, belief_score):
        """
        Negotiation Agent 职能：驱动 LLM 决策
        """

        # 调用您的 LLM API
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
        """
        信念更新机制：验证能否正确识别 Seller Type 。
        """
        if outcome["status"] == "accept":
            # 如果低价被接受，增加其为 flexible 的信心
            self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * 1.2)
        else:
            # 如果被拒绝，降低信心
            self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * 0.8)

    def run_step(self):
        """
        执行一个采样/谈判周期。
        """
        plan = self.orchestrator_plan()
        for seller_id, score in plan:
            if self.procured_units >= self.total_target:
                break

            # 执行谈判
            bid = self.negotiation_agent_action(seller_id, score)
            target_seller = next(s for s in self.sellers if s.seller_id == seller_id)
            result = target_seller.respond_to_bid(bid)

            # 更新信念与采购量
            self.update_belief(seller_id, result)
            if result["status"] == "accept":
                self.procured_units += 50  # 假设每轮采购50单位
                print(f"成功从 Seller {seller_id} 采购。当前总进度: {self.procured_units}/{self.total_target}")


# ==========================================
# 4. 运行 Simulation 1
# ==========================================
if __name__ == "__main__":
    llm_inference = GPTInference()
    # 创建 5 个卖方，模拟未知的市场构成
    market_sellers = [
        PassiveSeller(1, "hard"),
        PassiveSeller(2, "flexible"),
        PassiveSeller(3, "hard"),
        PassiveSeller(4, "flexible"),
        PassiveSeller(5, "flexible")
    ]

    buyer = BuyerSystem(market_sellers)

    print("开始 Simulation 1 — 建立收敛 Baseline...")
    for i in range(10):  # 运行10轮交互
        print(f"--- Round {i + 1} ---")
        buyer.run_step()
        print(f"当前信念分布: {buyer.beliefs}")