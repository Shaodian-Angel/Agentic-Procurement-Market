import random
from utils.gpt_inference import GPTInference


# ==========================================
# 1. 大语言模型接入点
# ==========================================
def call_your_llm(enhanced_seller_id, belief_score, procured_units) -> str:
    # 增加市场背景信息作为 Prompt 的一部分
    # 假设 llm_inference 已经支持传入新的 context
    response = llm_inference.generate_answer(
        enhanced_seller_id,
        belief_score,
        procured_units,
    )
    return response


# ==========================================
# 2. 卖方系统 (Passive Sellers)
# ==========================================
class PassiveSeller:
    def __init__(self, seller_id, seller_type):
        self.seller_id = seller_id
        self.type = seller_type
        self.reservation_price = 150 if seller_type == "hard" else 110

    def respond_to_bid(self, bid_price):
        if bid_price >= self.reservation_price:
            return {"status": "accept", "price": bid_price}
        else:
            return {"status": "reject", "counter_offer": self.reservation_price + 10}


# ==========================================
# 3. 买方 Multi-Agent 系统 (支持 Simulation 2)
# ==========================================
class BuyerSystemSim2:
    def __init__(self, sellers, total_target=1000):
        self.total_target = total_target
        self.procured_units = 0
        self.sellers = sellers
        self.beliefs = {s.seller_id: 0.5 for s in sellers}

        # --- Sim 2 新增：二级信念 (Second-order Belief) ---
        # 初始认为 Flexible 卖家在市场中的占比概率
        self.market_composition_belief = 0.5

    def update_market_belief(self):
        """
        Simulation 2 核心：根据所有个体信念更新对市场整体构成的判断 。
        """
        if not self.beliefs:
            return
        # 简单的二级更新逻辑：市场整体灵活度 = 个体信念的平均值
        self.market_composition_belief = sum(self.beliefs.values()) / len(self.beliefs)

    def orchestrator_plan(self):
        """
        Orchestrator 职能：利用二级信念调整全局策略 [cite: 12, 21]。
        """
        self.update_market_belief()

        # 根据市场整体特征调整行为：
        # 如果市场被认为是 "Hard" 为主 (belief < 0.4)，可能需要采取更保守或更广泛探测的策略
        strategy_context = "Aggressive" if self.market_composition_belief > 0.6 else "Probing"

        sorted_sellers = sorted(self.beliefs.items(), key=lambda x: x[1], reverse=True)
        return sorted_sellers, strategy_context

    # 和simulation1相比，这里增加了市场二级信念作为参数传入 negotiation_agent_action 函数
    def negotiation_agent_action(self, seller_id, belief_score, strategy_context):
        """
        修正后的 Negotiation Agent：将市场二级信念注入到 Seller 信息中
        """
        # 将市场整体情况合并到描述中，而不是作为一个独立的关键字参数
        # 这样就不需要修改 generate_answer 函数签名
        enhanced_seller_id = (
            f"{seller_id} (Market context: {strategy_context}, "
            f"Overall Market Flexibility: {self.market_composition_belief:.2f})"
        )

        # 调用时只传入原有的三个参数
        response = call_your_llm(enhanced_seller_id, belief_score, self.procured_units)

        try:
            import re
            match = re.search(r"FINAL_BID:\s*(\d+)", response)
            if match:
                return int(match.group(1))
        except:
            pass
        return 130

    def update_individual_belief(self, seller_id, outcome):
        if outcome["status"] == "accept":
            self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * 1.2)
        else:
            self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * 0.8)

    def run_step(self):
        plan, strategy_context = self.orchestrator_plan()
        print(f"[Orchestrator] Market Belief: {self.market_composition_belief:.2f}, Mode: {strategy_context}")

        for seller_id, score in plan:
            if self.procured_units >= self.total_target:
                break

            bid = self.negotiation_agent_action(seller_id, score, strategy_context)
            target_seller = next(s for s in self.sellers if s.seller_id == seller_id)
            result = target_seller.respond_to_bid(bid)

            self.update_individual_belief(seller_id, result)
            if result["status"] == "accept":
                self.procured_units += 50
                print(f"成功从 Seller {seller_id} 采购。进度: {self.procured_units}/{self.total_target}")


# ==========================================
# 4. 运行 Simulation 2
# ==========================================
if __name__ == "__main__":
    llm_inference = GPTInference()

    # 模拟场景 A：Hard 卖家占多数的市场 (Simulation 2 的变化特征 )
    print("\n--- Testing Sim 2: Hard-dominant Market ---")
    hard_market = [
        PassiveSeller(1, "hard"), PassiveSeller(2, "hard"),
        PassiveSeller(3, "hard"), PassiveSeller(4, "hard"),
        PassiveSeller(5, "flexible")
    ]
    buyer_a = BuyerSystemSim2(hard_market)
    for i in range(5):
        buyer_a.run_step()

    # 模拟场景 B：Flexible 卖家占多数的市场
    print("\n--- Testing Sim 2: Flexible-dominant Market ---")
    flex_market = [
        PassiveSeller(1, "flexible"), PassiveSeller(2, "flexible"),
        PassiveSeller(3, "flexible"), PassiveSeller(4, "flexible"),
        PassiveSeller(5, "hard")
    ]
    buyer_b = BuyerSystemSim2(flex_market)
    for i in range(5):
        buyer_b.run_step()