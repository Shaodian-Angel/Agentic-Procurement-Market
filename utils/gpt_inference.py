import re
import time
from openai import OpenAI


class GPTInference:
    def __init__(self):
        try:
            # self.client = OpenAI(
            #     api_key=AZURE_OPENAI_KEY,  # 这里填入令牌
            #     # base_url=""  # 注意：结尾只写到 /v1
            #     http_client=http_client  # 将代理交给 http_client 处理
            # )
            self.client = OpenAI(
                api_key="",
                base_url="",
            )
        except Exception as e:
            print(f"Error during API request: {str(e)}")
            print(f"Response: {getattr(e, 'response', 'No response')}")

    def predict(self, prompt):
        message_text = [
            {"role": "user", "content": prompt},
        ]
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-oss-120b-250805",  # 替换为你在 Ark 平台上的模型 ID,  # model = "deployment_name"
                    messages=message_text,
                    temperature=0,
                    max_tokens=1024,
                    seed=12345,
                )
                break
            except Exception as e:
                print(f"Error during API request: {str(e)}")
                print(f"Response: {getattr(e, 'response', 'No response')}")
                time.sleep(5)
        answer = self.post_process(response.choices[0].message.content)
        return answer

    def post_process(self, text):
        match = re.search(r"(?i)(?<=\banswer:\s).*", text)
        if match:
            return match.group(0)
        else:
            return text

    def generate_answer(self, seller_id, belief_score, procured_units):
        prompt = (f"""
        ### Role
        You are a professional Procurement Negotiation Agent.

        ### Context
        - Target Seller ID: {seller_id}
        - Belief Score (Prob of being Flexible): {belief_score:.2f}
        - Progress: {procured_units}/1000 units

        ### Strategy Guidance
        - Flexible sellers: Range [110, 130]
        - Hard bargainers: Range [150, 170]
        - Your goal is to minimize total cost while reaching 1000 units.

        ### Task
        Determine the bid price. 
        If the score is low (~0.1), consider if it's worth bidding higher or moving on.
        If the score is high (~0.9), be aggressive to save cost.

        Ending format: FINAL_BID: [price]
        """)
        answer = self.predict(prompt)
        return answer