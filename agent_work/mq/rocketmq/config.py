# rocketmq_config.py
import os
import glob
import jpype.imports

# 指定 JVM 路径
JVM_PATH = r"C:\Program Files\Java\jdk-1.8\jre\bin\server\jvm.dll"  # 请替换为你的 jvm.dll 路径

# 指定 RocketMQ JAR 包目录
LIB_DIR = r"C:\englishprogram\rocketmq-all-4.9.6-bin-release\lib"

# 获取所有 JAR 文件
jar_files = glob.glob(os.path.join(LIB_DIR, "*.jar"))
classpath_str = os.pathsep.join(jar_files)

# 启动 JVM
if not jpype.isJVMStarted():
    jpype.startJVM(JVM_PATH, "-Djava.class.path=" + classpath_str)

print("[成功] RocketMQ JAR 加载完成")


ROCKETMQ_CONFIG = {
    "namesrv_addr": "localhost:9876",
    "topic": "order_topic",
    "producer_group": "order_producer_group",
    "consumer_group": "order_consumer_group"
}

# 幂等性存储
processed_orders = set()