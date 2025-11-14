"""
    模拟多用户并发请求场景，启动方式：
    # Web UI 模式
    locust -f locustfile.py --web-port=8089

    # 命令行模式（50并发用户，持续5分钟）
    locust -f locustfile.py --headless -u 50 -r 5 -t 5m
"""
import random
from locust import FastHttpUser, task, between, tag

# 测试用问题池（覆盖运维场景，可按需扩展）
FIRST_QUESTIONS = [
    "M-2000报警E101怎么办？",
    "M-3000报警E056是什么问题？",
    "设备开机前要检查什么？",
    "主轴轴承多久保养一次？",
    "冷却滤芯更换周期？",
    "M-2000最大加工转速是多少？",
    "液压油多久换一次？",
    "变频器怎么校准？"
]

FOLLOW_UP_QUESTIONS = [
    "冷却液正常，下一步查什么？",
    "密封圈怎么检查？",
    "润滑脂用什么型号？",
    "检查需要断电吗？",
    "还是报错怎么办？"
]

class MaintenanceAPITester(FastHttpUser):
    """模拟运维用户并发请求"""
    host = "http://localhost:8090"  # 你的API服务地址（默认8000端口）
    wait_time = between(1, 3)  # 用户两次请求间隔1-3秒（贴近真实操作）
    session_id = None  # 存储当前用户的会话ID（多轮对话关联）

    @tag("首轮对话")
    @task(5)  # 权重5（出现概率最高，模拟多数用户首次咨询）
    def first_query(self):
        """模拟用户首轮咨询（无session_id，服务端自动创建）"""
        if not self.session_id:  # 仅首轮执行
            question = random.choice(FIRST_QUESTIONS)
            # 发送POST请求
            response = self.client.post(
                "/query",
                json={"question": question},
                headers={"Content-Type": "application/json"}
            )
            # 解析响应，获取session_id（用于后续多轮对话）
            if response.status_code == 200:
                self.session_id = response.json()["session_id"]
                print(f"会话{self.session_id}首轮成功")
            else:
                print(f"会话{self.session_id}首轮失败，状态码={response.status_code}")

    @tag("多轮对话")
    @task(2)  # 权重2（出现概率较低，模拟部分用户继续咨询）
    def follow_up_query(self):
        """模拟多轮对话（携带已获取的session_id）"""
        if self.session_id:  # 仅首轮成功后执行
            question = random.choice(FOLLOW_UP_QUESTIONS)
            response = self.client.post(
                "/query",
                json={"session_id": self.session_id, "question": question},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                print(f"会话{self.session_id}多轮失败，状态码={response.status_code}")

    @tag("混合场景")
    @task(3)  # 权重3（混合首轮/多轮，更贴近真实流量）
    def mixed_query(self):
        """随机发起首轮或多轮对话"""
        if random.random() < 0.3 or not self.session_id:
            self.first_query()  # 30%概率首轮，或无session_id时
        else:
            self.follow_up_query()  # 70%概率多轮