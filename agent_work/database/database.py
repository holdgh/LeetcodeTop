import pytz
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, CheckConstraint, Integer, Index, Enum
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum
import datetime
import os

# 数据库配置（可替换为环境变量）
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_memory")

# 基础模型
Base = declarative_base()


# 1. 用户表（若已有用户系统可复用，此处为简化版）
class User(Base):
    __tablename__ = "users"
    user_id = Column(String(64), primary_key=True, comment="用户唯一标识（如登录态ID）")
    create_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
                         comment="创建时间")
    # 关联会话
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


# 2. 会话表（对应session_id，关联用户）
class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String(64), primary_key=True, comment="会话ID（与现有session_id一致）")
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, comment="关联用户ID（用户隔离核心）")
    create_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
                         comment="会话创建时间")
    update_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
                         onupdate=datetime.datetime.now(datetime.timezone.utc),
                         comment="最后更新时间")
    # 关联对话
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    # 关联用户
    user = relationship("User", back_populates="sessions")


# 3. 对话消息表（存储每轮对话内容）
class Message(Base):
    __tablename__ = "messages"
    id = Column(String(64), primary_key=True, comment="消息唯一ID（UUID）")
    session_id = Column(String(64), ForeignKey("sessions.session_id"), nullable=False, comment="关联会话ID")
    conversation_id = Column(String(64), nullable=False, comment="对话ID（与现有conversation_id一致）")
    role = Column(String(20), nullable=False,
                  comment="消息角色：user-用户，retriever-检索助手，expert-运维专家，system-系统，rewriter-重写助手")
    content = Column(Text, nullable=False, comment="消息内容（文本）")
    timestamp = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
                       comment="消息时间戳")
    # 关联会话
    session = relationship("Session", back_populates="messages")

    """
    对应数据表侧执行下述sql：
    -- 假设 messages 表已存在，修改 role 字段
    ALTER TABLE messages 
    ADD CONSTRAINT valid_role CHECK (
        role IN ('user', 'retriever', 'expert', 'system')
    );
    """
    # 添加表级 CHECK 约束（如果在 SQL 中未手动添加）
    __table_args__ = (
        CheckConstraint(
            role.in_(["user", "retriever", "expert", "system", "rewriter"]),
            name="valid_role"
        ),
    )


class MessageSummary(Base):
    """
    消息摘要表模型
    用于存储每个会话的最新摘要，以优化长对话历史的加载速度。
    """
    __tablename__ = "message_summaries"

    session_id = Column(
        String(64),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        primary_key=True,
        comment="关联会话ID"
    )
    latest_summary = Column(
        Text,
        comment="会话最新摘要（压缩长对话历史，减少加载耗时）"
    )
    summary_time = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.datetime.now(datetime.timezone.utc),
        comment="最后一次摘要生成时间"
    )
    total_messages = Column(
        Integer,
        nullable=False,
        default=0,
        comment="会话总消息数"
    )
    # 新增字段：上次摘要已处理的对话数（核心）
    last_processed_conversation_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="上次摘要已处理的对话数"
    )

    def __repr__(self):
        return f"<MessageSummary(session_id='{self.session_id}', summary_time='{self.summary_time}')>"


# ---------------------- 1. 定义PostgreSQL枚举类型（映射数据库枚举） ----------------------
# class TempMessageStatus(enum.Enum):
#     """临时消息状态枚举（和数据库temp_message_status枚举完全一致）"""
#     PENDING = "pending"  # 待恢复（Redis入队失败，兜底写入）
#     RECOVERED = "recovered"  # 已恢复（已重新入Redis队列）
#     TIMEOUT = "timeout"  # 已超时（超过阈值未恢复，标记为过期）


class TempMessageStatus(str, Enum):
    """temp_message表统一状态（覆盖全链路兜底）"""
    # 问答服务→Redis环节
    REDIS_FAILED = "redis_failed"  # 写入Redis失败，兜底落库
    REDIS_SUCCESS = "redis_success"  # 写入Redis成功
    # Redis→MQ环节
    MQ_PENDING = "mq_pending"      # 从Redis取出，待发MQ
    MQ_FAILED = "mq_failed"        # 发送MQ失败
    MQ_SENT = "mq_sent"            # 发送MQ成功
    # MQ→数据库环节
    DB_FAILED = "db_failed"        # 入库失败（重试耗尽）
    DB_SUCCESS = "db_success"      # 入库成功（最终态）
    # 通用状态
    TIMEOUT = "timeout"            # 超时未处理（清理脚本用）


# ---------------------- 2. temp_message表模型定义（和User表风格一致） ----------------------
class TempMessage(Base):
    __tablename__ = "temp_message"
    __table_args__ = (
        # 核心：添加检查约束，确保status值在枚举范围内（和数据库枚举对齐）
        CheckConstraint(
            "status IN ('redis_failed', 'redis_success', 'timeout', 'mq_pending', 'mq_failed', 'mq_sent', 'db_failed', 'db_success')",
            name="ck_temp_message_status"
        ),
        # 唯一索引：session_id + message_type（避免重复兜底写入）
        Index("idx_temp_message_session_type", "session_id", "message_type", unique=True),
        # 普通索引：status + backup_time（优化补偿任务查询）
        Index("idx_temp_message_status_backup_time", "status", "backup_time"),
        # 普通索引：updated_at（优化清理脚本查询）
        Index("idx_temp_message_updated_at", "updated_at"),
        {"comment": "Redis兜底临时消息表（Redis挂掉时缓存消息，避免数据丢失）"}
    )

    # 核心字段（和数据库表一一对应，注释清晰）
    id = Column(String(64), primary_key=True, comment="消息唯一标识，主键")
    session_id = Column(String(64), nullable=False, comment="会话ID（关联问答会话）")
    user_id = Column(String(64), nullable=False, comment="用户ID（关联问答会话）")
    message_type = Column(String(32), nullable=False, comment="消息类型：user/rewrite/retrieve/expert")
    content = Column(Text, nullable=False, comment="消息内容（Redis兜底核心数据）")
    generate_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
                           comment="消息生成时间戳（秒级，和原有message表对齐）")
    backup_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
                         comment="兜底写入时间戳（记录何时写入临时表）")
    # status = Column(
    #     Enum(TempMessageStatus, name="temp_message_status"),  # 映射PostgreSQL枚举类型
    #     nullable=False,
    #     default=TempMessageStatus.REDIS_FAILED,
    #     comment="消息状态：pending-待恢复/recovered-已恢复/timeout-已超时"
    # )
    # 核心修复：改用String类型，长度设为枚举值的最大长度（12）
    status = Column(
        String(12),  # 所有枚举值中最长的是"db_success"（10），留余量设12
        nullable=False,
        default=TempMessageStatus.REDIS_FAILED,  # 直接赋值字符串
        comment="消息状态（全链路兜底）"
    )
    created_at = Column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
        comment="数据库写入时间（带时区，便于跨时区排查）"
    )
    updated_at = Column(
        DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc),
        onupdate=datetime.datetime.now(datetime.timezone.utc),
        comment="状态更新时间（带时区，自动更新）"
    )

    # 代码层枚举校验（保证写入值合法）
    @property
    def status_enum(self):
        """将字符串转为枚举实例，方便代码使用"""
        return TempMessageStatus(self.status)

    @status_enum.setter
    def status_enum(self, value: TempMessageStatus):
        """赋值时用枚举实例，自动转为字符串"""
        self.status = value.value

    def __repr__(self):
        """自定义打印格式，便于调试"""
        return f"<TempMessage(id={self.id}, session_id='{self.session_id}', message_type='{self.message_type}', status='{self.status.value}')>"

    def to_dict(self):
        """序列化，避免枚举值报错"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_type": self.message_type,
            "content": self.content,
            "generate_time": self.generate_time,
            "backup_time": self.backup_time,
            "status": self.status.value,  # 取枚举值（字符串）
            "retry_count": self.retry_count,
            "error_msg": self.error_msg,
            "created_at": self.created_at if self.created_at else None,
            "updated_at": self.updated_at if self.updated_at else None
        }


# 异步数据库引擎初始化
engine = create_async_engine(DATABASE_URL, echo=False, pool_size=20, max_overflow=30)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# 数据库初始化函数（首次运行时执行）
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("数据库表创建成功！")


# 获取异步数据库会话
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()
