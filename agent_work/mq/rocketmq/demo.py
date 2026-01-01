if __name__ == '__main__':
    # 加载配置
    import jpype
    import jpype.imports# 指定 JVM 路径
    import os
    import glob
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
    from pyrocketmq import *
    # do something
    # jpype.shutdownJVM()
    # 消息生产者
    import json
    from pyrocketmq.common.message import Message
    from pyrocketmq.client.producer import Producer, SendStatus

    pr = Producer('test_producer')
    pr.setNamesrvAddr('localhost:9876')
    pr.start()
    body = json.dumps({'name': 'Alice', 'age': 1}).encode('utf-8')
    msg = Message(topic='test_topic', body=body, tags='girl')
    # send, tcp-like, return sendStatus
    sr = pr.send(msg)
    assert (sr.sendStatus == SendStatus.SEND_OK)
    pr.shutdown()
    # 消息消费者
    import json
    from pyrocketmq.client.consumer.consumer import PullConsumer, PullStatus

    cs = PullConsumer('test_pull_consumer')
    cs.setNamesrvAddr('localhost:9876')
    topic = 'test_topic'
    cs.start()
    # pull messages from each queue
    mqs = cs.fetchSubscribeMessageQueues(topic)
    for mq in mqs:
        ofs = cs.minOffset(mq)
        pr = cs.pull(mq, subExpression='girl', offset=ofs, maxNums=1)
        if pr.pullStatus == PullStatus.FOUND:
            # iterate msg in pull result
            for msg in pr:
                print(json.loads(msg.body))
    cs.shutdown()
    # 消费者
    import json
    import time
    from typing import List
    from pyrocketmq.client.consumer.listener import ConsumeConcurrentlyContext, ConsumeConcurrentlyStatus, \
        MessageListenerConcurrently
    from pyrocketmq.client.consumer.consumer import MessageSelector, PushConsumer
    from pyrocketmq.common.common import ConsumeFromWhere
    from pyrocketmq.common.message import MessageExt


    # subclass MessageListenerConcurrently to write your own consume action
    class MyMessageListenerConcurrently(MessageListenerConcurrently):
        def _consumeMessage(self, msgs: List[MessageExt],
                            context: ConsumeConcurrentlyContext) -> ConsumeConcurrentlyStatus:
            print('Concurrently', context.ackIndex)
            for msg in msgs:
                print(json.loads(msg.body))
            return ConsumeConcurrentlyStatus.CONSUME_SUCCESS


    cs = PushConsumer('test_push_consumer')
    cs.setNamesrvAddr('localhost:9876')
    selector = MessageSelector.byTag('girl')
    ml = MyMessageListenerConcurrently()
    cs.registerMessageListener(ml)
    cs.subscribe('test_topic', selector)
    cs.setConsumeFromWhere(ConsumeFromWhere.CONSUME_FROM_FIRST_OFFSET)
    cs.start()
    time.sleep(5)
    cs.shutdown()

    jpype.shutdownJVM()