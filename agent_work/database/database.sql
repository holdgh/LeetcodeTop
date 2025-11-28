-- ----------------------------
-- 1. 用户表（存储用户基础信息）
-- ----------------------------
CREATE TABLE "users" (
  "user_id" VARCHAR(64) NOT NULL PRIMARY KEY,
  "create_time" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "update_time" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 为用户表添加注释
COMMENT ON TABLE "users" IS '存储用户基础信息';
COMMENT ON COLUMN "users"."user_id" IS '用户唯一标识（如登录态ID、OpenID）';
COMMENT ON COLUMN "users"."create_time" IS '用户创建时间';
COMMENT ON COLUMN "users"."update_time" IS '用户信息最后更新时间';

-- 索引：用户ID唯一索引（主键已默认创建，此处为冗余示例，可省略）
CREATE UNIQUE INDEX "idx_user_id" ON "users" USING btree ("user_id");

-- ----------------------------
-- 2. 会话表（关联用户，存储会话上下文）
-- ----------------------------
CREATE TABLE "sessions" (
  "session_id" VARCHAR(64) NOT NULL PRIMARY KEY,
  "user_id" VARCHAR(64) NOT NULL,
  "create_time" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "update_time" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "status" SMALLINT NOT NULL DEFAULT 1,
  CONSTRAINT "fk_session_user" FOREIGN KEY ("user_id") REFERENCES "users" ("user_id") ON DELETE CASCADE
);

-- 为会话表添加注释
COMMENT ON TABLE "sessions" IS '关联用户，存储会话上下文';
COMMENT ON COLUMN "sessions"."session_id" IS '会话唯一ID（与业务层session_id一致）';
COMMENT ON COLUMN "sessions"."user_id" IS '关联用户ID（用户隔离核心）';
COMMENT ON COLUMN "sessions"."create_time" IS '会话创建时间';
COMMENT ON COLUMN "sessions"."update_time" IS '会话最后更新时间（最后一次对话时间）';
COMMENT ON COLUMN "sessions"."status" IS '会话状态：1-活跃，0-已关闭';

-- 索引：用户ID+会话状态复合索引（优化用户会话查询）
CREATE INDEX "idx_session_user_status" ON "sessions" USING btree ("user_id", "status");
-- 索引：更新时间索引（优化会话清理、排序查询）
CREATE INDEX "idx_session_update_time" ON "sessions" USING btree ("update_time");

-- ----------------------------
-- 3. 对话消息表（存储单轮对话内容）
-- ----------------------------
CREATE TABLE "messages" (
  "id" VARCHAR(64) NOT NULL PRIMARY KEY,
  "session_id" VARCHAR(64) NOT NULL,
  "conversation_id" VARCHAR(64) NOT NULL,
  "role" VARCHAR(20) NOT NULL,
  "content" TEXT NOT NULL,
  "timestamp" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "metadata" JSONB,
  CONSTRAINT "fk_message_session" FOREIGN KEY ("session_id") REFERENCES "sessions" ("session_id") ON DELETE CASCADE
);

-- 为对话消息表添加注释
COMMENT ON TABLE "messages" IS '存储单轮对话内容';
COMMENT ON COLUMN "messages"."id" IS '消息唯一ID（UUID格式）';
COMMENT ON COLUMN "messages"."session_id" IS '关联会话ID';
COMMENT ON COLUMN "messages"."conversation_id" IS '关联对话ID（单轮对话标识）';
COMMENT ON COLUMN "messages"."role" IS '消息角色：user-用户，retriever-检索助手，expert-运维专家，system-系统';
COMMENT ON COLUMN "messages"."content" IS '消息内容（文本格式）';
COMMENT ON COLUMN "messages"."timestamp" IS '消息发送时间';
COMMENT ON COLUMN "messages"."metadata" IS '消息元数据（可选，存储额外信息如工具调用参数、消息类型等）';

-- 索引：会话ID+时间戳复合索引（优化会话内消息查询、排序）
CREATE INDEX "idx_message_session_timestamp" ON "messages" USING btree ("session_id", "timestamp");
-- 索引：对话ID索引（优化单轮对话消息聚合查询）
CREATE INDEX "idx_message_conversation_id" ON "messages" USING btree ("conversation_id");
-- 索引：角色索引（优化按角色筛选消息）
CREATE INDEX "idx_message_role" ON "messages" USING btree ("role");

-- ----------------------------
-- 4. 消息摘要表（可选，用于优化长对话历史加载速度）
-- ----------------------------
CREATE TABLE "message_summaries" (
  "session_id" VARCHAR(64) NOT NULL PRIMARY KEY,
  "latest_summary" TEXT,
  "summary_time" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "total_messages" INTEGER NOT NULL DEFAULT 0,
  CONSTRAINT "fk_summary_session" FOREIGN KEY ("session_id") REFERENCES "sessions" ("session_id") ON DELETE CASCADE
);

-- 为消息摘要表添加注释
COMMENT ON TABLE "message_summaries" IS '用于优化长对话历史加载速度';
COMMENT ON COLUMN "message_summaries"."session_id" IS '关联会话ID';
COMMENT ON COLUMN "message_summaries"."latest_summary" IS '会话最新摘要（压缩长对话历史，减少加载耗时）';
COMMENT ON COLUMN "message_summaries"."summary_time" IS '最后一次摘要生成时间';
COMMENT ON COLUMN "message_summaries"."total_messages" IS '会话总消息数';

-- 索引：摘要时间索引（优化摘要更新、清理逻辑）
CREATE INDEX "idx_summary_time" ON "message_summaries" USING btree ("summary_time");

-- ----------------------------
-- 5. 分区表配置（可选，适用于消息量巨大场景）
-- ----------------------------
-- 1. 创建分区父表（主键包含分区键 timestamp）
--CREATE TABLE "messages_partitioned" (
--  "id" VARCHAR(64) NOT NULL,
--  "session_id" VARCHAR(64) NOT NULL,
--  "conversation_id" VARCHAR(64) NOT NULL,
--  "role" VARCHAR(20) NOT NULL,
--  "content" TEXT NOT NULL,
--  "timestamp" TIMESTAMP WITH TIME ZONE NOT NULL,
--  "metadata" JSONB,
--  -- 主键包含 id 和 timestamp（分区键）
--  PRIMARY KEY ("id", "timestamp"),
--  CONSTRAINT "fk_message_session_partitioned" FOREIGN KEY ("session_id") REFERENCES "sessions" ("session_id") ON DELETE CASCADE
--) PARTITION BY RANGE ("timestamp");
--
---- 为分区父表添加注释
--COMMENT ON TABLE "messages_partitioned" IS '按时间分区的对话消息表（适用于消息量巨大场景）';
---- 其他列的注释可以按需添加...
--
---- 2. 创建初始分区（示例：2024年Q1、Q2分区）
--CREATE TABLE "messages_2024_q1" PARTITION OF "messages_partitioned"
--FOR VALUES FROM ('2024-01-01 00:00:00') TO ('2024-04-01 00:00:00');
--
--CREATE TABLE "messages_2024_q2" PARTITION OF "messages_partitioned"
--FOR VALUES FROM ('2024-04-01 00:00:00') TO ('2024-07-01 00:00:00');

---- 3. 为分区创建索引（可选，PostgreSQL有时会自动继承）
--CREATE INDEX "idx_message_partitioned_session_timestamp" ON "messages_2024_q1" USING btree ("session_id", "timestamp");
--CREATE INDEX "idx_message_partitioned_session_timestamp" ON "messages_2024_q2" USING btree ("session_id", "timestamp");


-- 4. 后续可通过定时任务自动创建新分区（如每年12月创建下一年的4个季度分区）