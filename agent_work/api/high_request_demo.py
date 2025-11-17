"""
    模拟多用户并发请求场景，启动方式：
    # Web UI 模式
    locust -f locustfile.py --web-port=8091

    # 命令行模式（50并发用户，持续5分钟）
    locust -f locustfile.py --headless -u 50 -r 5 -t 5m
"""
import random
from locust import FastHttpUser, task, between, tag


class MaintenanceAPITester(FastHttpUser):
    """模拟运维用户并发请求"""
    host = "http://localhost:8081"  # 你的API服务地址（默认8000端口）
    wait_time = between(1, 3)  # 用户两次请求间隔1-3秒（贴近真实操作）
    order_no = None  # 订单编号

    @tag("首轮对话")
    @task(5)  # 权重5（出现概率最高，模拟多数用户首次咨询）
    def first_query(self):
        """模拟用户下单"""
        # 发送POST请求
        response = self.client.post(
            "/api/work",
            headers={"Content-Type": "application/json"}
        )
        # 解析响应，获取session_id（用于后续多轮对话）
        if response.status_code == 200:
            self.order_no = response.json()["order_no"]
            print(f"订单{self.order_no}首轮成功")
        else:
            print(f"订单创建失败，状态码={response.status_code}")
