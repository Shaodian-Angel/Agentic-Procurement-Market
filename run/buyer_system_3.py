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
                 lambda_threshold=0.01,  # 对微小让步的过度反应 # 0.01-0.02
                 delta_ask=0.1,  # PushBack 时的加价幅度减小，避免价格剧烈震荡 # 改为百分比更好，10%或更大会更好
                 gamma=0.1,  # 改为百分比，和delta_ask之间的差距更大一些？
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
        if self.interaction_count < 5:
            # 前几轮数据不足时，放宽条件：至少1次出价 >= ask-5
            return any(b >= self.current_ask - 5 for b in self.bid_history)
        recent_bids = self.bid_history[-10:]
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

        # 记录 Seller 状态日志
        self.seller_states_log = {s.seller_id: [] for s in sellers}

        # [新增] 记录每次具体的对话历史
        self.interaction_history = []

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

                # [新增] 解析并记录本次对话信息
                if result["status"] == "accept":
                    seller_ask = target_seller.current_ask
                    trade_price = result["price"]
                    trade_qty = 50
                    self.total_spent += (result["price"] * 50)
                    self.procured_units += 50
                    # 成功交易，提升对该卖家的信念
                    self.beliefs[seller_id] = min(0.99, self.beliefs[seller_id] * (1 + penalty))
                else:
                    seller_ask = result["counter_offer"]
                    trade_price = 0  # 无交易则为0
                    trade_qty = 0  # 无交易则为0
                    self.beliefs[seller_id] = max(0.01, self.beliefs[seller_id] * (1 - penalty))

                # [新增] 将数据添加到本次运行的对话历史列表中
                self.interaction_history.append({
                    "seller_id": seller_id,  # 记录卖家的 ID
                    "bid": bid,
                    "ask": seller_ask,
                    "trade_price": trade_price,
                    "trade_qty": trade_qty
                })

                self.seller_states_log[seller_id].append(result['state'])
                if 'counter_offer' in result:
                    round_asks.append(result['counter_offer'])

            if round_asks:
                self.round_ask_prices.append(np.mean(round_asks))
            else:
                self.round_ask_prices.append(0)

        return rounds


# ==========================================
# 3. 实验引擎 (Batch Experiment)
# ==========================================
def run_batch_experiment(num_runs=200, rounds_limit=20, penalty=0.2, lambda_threshold=0.01,  # 从 0.08 降至 0.02。减少对微小让步的过度反应
                 delta_ask=0.1,  # 从 7 降至 5。PushBack 时的加价幅度减小，避免价格剧烈震荡
                 gamma=0.05,  # 让步幅度保持不变
                 discount_rate=0.1):
    results = []
    detailed_history_data = []

    output_dir = "result/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # [新增] 预先定义好详细对话记录的文件路径
    detailed_csv_name = f"{output_dir}/detailed_dialog/sim3_fsm_detailed_dialog_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_{gamma}_gamma.csv"

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

        # 提取并构建当前轮次的详细对话数据
        initial_expected_price = buyer.interaction_history[0]["bid"] if buyer.interaction_history else 0

        row_detailed = {
            "times": i,
            "market_type": m_type,
            "buyers_initial_expected_price": initial_expected_price
        }

        # 动态生成 “第N轮对话” 的列内容
        for idx, interaction in enumerate(buyer.interaction_history):
            record_str = f"{{buyer_bids:{interaction['bid']:.2f}, seller_ID:{interaction['seller_id']}, seller_bids:{interaction['ask']:.2f}, trade_price:{interaction['trade_price']:.2f}, trade_qty:{interaction['trade_qty']}}}"
            row_detailed[f"Communication Round {idx + 1}"] = record_str

        detailed_history_data.append(row_detailed)

        # ==========================================
        # [新增核心逻辑] 每次 Run 结束后立刻进行数据保存防呆处理
        # ==========================================
        df_temp = pd.DataFrame(detailed_history_data)

        # 提取列名并对动态生成的“第N轮对话”进行数字升序排序
        fixed_cols = ["times", "market_type", "buyers_initial_expected_price"]
        dialog_cols = [c for c in df_temp.columns if c.startswith("Communication Round")]
        dialog_cols = sorted(dialog_cols, key=lambda x: int(re.search(r'\d+', x).group()))

        # 重组 DataFrame 确保列序正确，然后直接覆写保存
        df_temp = df_temp[fixed_cols + dialog_cols]
        df_temp.to_csv(detailed_csv_name, index=False, encoding='utf-8-sig')

        # --- 收集宏观统计数据 ---
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

        # 2. Convergence Round (连续3轮 state 不变 且 ask 变化率 < 0.5%) # 感觉这个判断不太正确
        # ==========================================
        # 核心定义：结束前连续 >= 5 次互动均与同一卖家进行，且期间价格波动率极小
        # ==========================================
        # convergence_round = rounds_limit + 1
        # history = buyer.interaction_history
        # logs = buyer.seller_states_log
        #
        # if len(history) >= 5:
        #     # 1. 从最后一次交易往前回溯，看看买家最终“锁定”了哪个卖家
        #     last_seller_id = history[-1]['seller_id']
        #
        #     # 2. 往前数，找到连续与这个卖家对话的起点
        #     streak_start_idx = len(history) - 1
        #     while streak_start_idx > 0 and history[streak_start_idx - 1]['seller_id'] == last_seller_id:
        #         streak_start_idx -= 1
        #
        #     streak_length = len(history) - streak_start_idx
        #
        #     # 3. 判断：是否连续缠着这一个卖家超过 5 回合？
        #     if streak_length >= 5:
        #         # 截取这最后一段稳定的对话记录
        #         streak_records = history[streak_start_idx:]
        #
        #         # 4. 判断：这期间的价格波动是否极小？
        #         bids = [r['bid'] for r in streak_records]
        #         trade_price = [r['trade_price'] for r in streak_records]
        #
        #         # 计算这段时间内最高价与最低价的波动比例
        #         bid_fluctuation = (max(bids) - min(bids)) / min(bids) if min(bids) > 0 else 0
        #         trade_price_fluctuation = (max(trade_price) - min(trade_price)) / min(trade_price) if min(trade_price) > 0 else 0
        #
        #         # 设定阈值：如果买方出价和交易价格的上下波动幅度都小于 5%
        #         if bid_fluctuation <= 0.05 and trade_price_fluctuation <= 0.05:
        #             # 记录收敛轮次为这段稳定期的起点 (索引从0开始，所以+1转为轮次)
        #             convergence_round = streak_start_idx + 1

        # ==========================================
        # [修改] 重新定义的收敛判断逻辑 (基于全市场交易价格稳定)
        # 核心定义：结束前最后 5 次“成功交易”的价格波动率极小
        # ==========================================
        convergence_round = rounds_limit + 1
        history = buyer.interaction_history
        logs = buyer.seller_states_log

        # 1. 过滤出所有真正达成交易（交易价格 > 0 且 数量 > 0）的对话记录
        successful_trades = [r for r in history if r['trade_price'] > 0 and r['trade_qty'] > 0]

        # 2. 判断整个实验过程中，成功交易的次数是否至少有 5 次
        if len(successful_trades) >= 5:
            # 3. 截取最后 5 次成功交易的记录
            last_5_trades = successful_trades[-5:]
            trade_prices = [r['trade_price'] for r in last_5_trades]

            # 4. 计算这 5 次交易价格的最高价与最低价的波动比例
            max_price = max(trade_prices)
            min_price = min(trade_prices)
            price_fluctuation = (max_price - min_price) / min_price if min_price > 0 else 0

            # 5. 设定阈值：如果全市场最后 5 次交易的价格上下波动幅度小于 5%
            if price_fluctuation <= 0.05:
                # 寻找这 5 次成功交易里，“最早的那一次交易”在总对话历史（history）中的位置
                # 这样可以精准定位到系统从“第几轮对话”开始进入价格稳定期的
                first_stable_trade = last_5_trades[0]

                for idx, record in enumerate(history):
                    # 通过对比引用或关键字段，找到它在原总历史列表中的索引
                    if (record['bid'] == first_stable_trade['bid'] and
                            record['trade_price'] == first_stable_trade['trade_price']):
                        convergence_round = idx + 1  # 转换为轮次 (1-based)
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
            "market_distribution": ",".join(dist),
            "final_aup": aup,
            "rounds": rounds_taken,
            "final_market_belief": buyer.market_composition_belief,
            "focus_on_flexible_ratio": pref_ratio,
            "final_seller_state_distribution": state_dist_str,
            "convergence_round": convergence_round,
            "system_cycle_flag": system_cycle_flag,
            "total_procured_units": buyer.procured_units  # 记录买家最终总共达成的交易数量
        })

        if i % 10 == 0:
            print(f"进度: {i}/{num_runs} | 当前 AUP 均值: {pd.DataFrame(results)['final_aup'].mean():.2f}")
            # 同理，将宏观统计数据也实时保存一下防中断
            pd.DataFrame(results).to_csv(
                f"{output_dir}/sim3_fsm_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_{gamma}_gamma_intermediate_stats.csv",
                index=False)

    print(f"✅ 实验完成！每一次对话的历史明细已最终完整保存至: {detailed_csv_name}")
    return pd.DataFrame(results)


# ==========================================
# 4. 统计可视化分析
# ==========================================
def analyze_200_runs(df, detail, limit):
    plt.style.use('seaborn-v0_8-whitegrid')
    # [修改] 拓宽画布以容纳 5 张图
    plt.figure(figsize=(23, 5))

    # 1. AUP 密度分布 (分市场类型)
    plt.subplot(1, 5, 1)
    sns.histplot(data=df, x="final_aup", hue="market_type", kde=True, palette="husl")
    plt.title("AUP Distribution Density")

    # ==========================================
    # 2. 总体达成交易数量可视化 (Total Procured Quantity)
    # ==========================================
    plt.subplot(1, 5, 2)
    try:
        # 自动识别文件中类似于 "Communication Round X" 或 "第X轮对话" 的列
        dialog_cols = [c for c in detail.columns if "Round" in c or "轮" in c]
        success_rounds_data = []
        # 遍历每一行数据
        for _, row in detail.iterrows():
            m_type = row["market_type"] if "market_type" in row else row["市场环境"]

            for col in dialog_cols:
                cell_value = row[col]
                if pd.isna(cell_value):
                    continue

                # 检查字符串中是否有 trade_qty 并且其数量大于 0
                qty_match = re.search(r'trade_qty:(\d+)', str(cell_value))
                if qty_match:
                    trade_qty = int(qty_match.group(1))
                    # 只有真正达成交易(数量>0)的轮次才被计入分布
                    if trade_qty > 0:
                        # 提取出当前的轮次数字
                        round_num = int(re.search(r'\d+', col).group())

                        # 为了和直方图频数对齐，如果 trade_qty 是 50，我们可以视作达成了 1 次完整批次交易
                        # 或者直接把这轮数字塞进列表，让直方图统计它的“发生频次密度”
                        success_rounds_data.append({
                            "Successful_Round": round_num,
                            "market_type": m_type
                        })
        df_success = pd.DataFrame(success_rounds_data)
        # 绘制和图一完全一样风格的 histplot 密度分布图
        # multiple="layer" 代表像图一一样半透明重叠，kde=True 生成平滑密度曲线
        sns.histplot(data=df_success, x="Successful_Round", hue="market_type",
                     kde=True, element="step", multiple="layer", palette="husl", binwidth=1)

        plt.title("Trade Success Round Distribution")
        plt.xlabel("Communication Round")
        plt.ylabel("Count of Successful Trades")

    except Exception as e:
        plt.text(0.5, 0.5, f"Plot Failed:\n{str(e)}", ha='center', va='center')
        plt.title("Trade Success Round (Error)")
    plt.legend()

    # 3. 策略倾向性分析 (Flexible Interaction Ratio)
    plt.subplot(1, 5, 3)
    sns.boxplot(x="market_type", y="focus_on_flexible_ratio", data=df, palette="Pastel1")
    plt.title("Strategic Focus on Flexible Sellers")

    # 4. 学习效果：信念 vs 成本
    plt.subplot(1, 5, 4)
    # 定义两种市场环境及其对应的颜色（从 husl 调色盘提取，确保全图风格高度一致）
    # 通常 husl 的前两个主色是粉红/红（代表 Hard）和蓝/绿（代表 Flexible）
    market_types = ["Hard-dominant", "Flexible-dominant"]
    colors = {"Hard-dominant": "#f46d43", "Flexible-dominant": "#66c2a5"}

    for m_type in market_types:
        # 筛选出当前市场类型的数据
        sub_df = df[df['market_type'] == m_type]

        # 如果当前类型有数据，则绘制回归线
        if not sub_df.empty:
            sns.regplot(
                data=sub_df,
                x="final_market_belief",
                y="final_aup",
                scatter_kws={'alpha': 0.4},  # 散点半透明度，防止重叠遮挡
                line_kws={'linewidth': 2},  # 趋势线粗细
                color=colors[m_type],  # 绑定颜色
                label=f"{m_type} Trend",
                ax=plt.gca()  # 确保画在同一个子图里
            )
    plt.title("Learning Efficiency\n(Belief vs Cost)")
    plt.xlabel("Final Market Belief")
    plt.ylabel("Final AUP")
    plt.legend()

    # 5. 收敛轮数分布直方图
    plt.subplot(1, 5, 5)
    hard_converged_count = sum((df["market_type"] == "Hard-dominant") & (df["convergence_round"] <= limit))
    flexible_converged_count = sum((df["market_type"] == "Flexible-dominant") & (df["convergence_round"] <= limit))
    hard_convergence_rate = hard_converged_count / len(df["market_type"] == "Hard-dominant")
    flexible_convergence_rate = flexible_converged_count / len(df["market_type"] == "Flexible-dominant")

    sns.histplot(data=df, x="convergence_round", hue="market_type", multiple="stack", palette="coolwarm")
    plt.title(f"Convergence Round Distribution\n(Hard Market Rate: {hard_convergence_rate:.1%}; Felxible Market Rate: {flexible_convergence_rate:.1%})")
    plt.axvline(x=limit, color='black', linestyle='--', label='Limit')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"result/sim3_fsm_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_{gamma}_gamma_final_analysis.png")
    plt.show()


if __name__ == "__main__":
    # 运行 200 次
    llm = GPTInference()
    num_runs = 200
    rounds_limit = 50
    penalty = 0.3  # 选择hard卖家交易的惩罚力度
    lambda_threshold = 0.01  # 对微小让步的反应
    delta_ask = 0.1  # PushBack 时的加价幅度减小，避免价格剧烈震荡
    gamma = 0.15  # 让步幅度和delta_ask之间的差距大一些
    df_final = run_batch_experiment(num_runs, rounds_limit, penalty, lambda_threshold, delta_ask, gamma)

    # 保存最终详细结果
    df_final.to_csv(f"result/sim3_fsm_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_full_data.csv",
                    index=False)
    df_final = read_csv(
        f"result/sim3_fsm_final_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_full_data.csv")  # 重新读取以确保数据完整性
    df_detail = pd.read_csv(f"result/detailed_dialog/sim3_fsm_detailed_dialog_{num_runs}_runs_{rounds_limit}_rounds_{penalty}_penalty_{lambda_threshold}_lambda_{delta_ask}_delta_{gamma}_gamma.csv")
    # 分析
    analyze_200_runs(df_final, df_detail, rounds_limit)

    # 打印按市场类型分类的描述统计
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY BY MARKET TYPE (FSM SELLER)")
    print("=" * 50)
    print(df_final.groupby('market_type')[['final_aup', 'rounds', 'focus_on_flexible_ratio']].mean())
    print("\nState Distribution Samples:")
    print(df_final['final_seller_state_distribution'].value_counts().head())