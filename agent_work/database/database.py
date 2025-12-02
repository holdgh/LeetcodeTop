import pytz
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, CheckConstraint, Integer
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
    create_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc), comment="创建时间")
    # 关联会话
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


# 2. 会话表（对应session_id，关联用户）
class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String(64), primary_key=True, comment="会话ID（与现有session_id一致）")
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, comment="关联用户ID（用户隔离核心）")
    create_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc), comment="会话创建时间")
    update_time = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc), onupdate=datetime.datetime.now(datetime.timezone.utc),
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
    timestamp = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc), comment="消息时间戳")
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
