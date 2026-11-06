-- CreateEnum MessageRole
DO $$ BEGIN
    CREATE TYPE "MessageRole" AS ENUM ('USER', 'ASSISTANT', 'SYSTEM', 'TOOL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- CreateEnum MemoryType
DO $$ BEGIN
    CREATE TYPE "MemoryType" AS ENUM ('SEMANTIC', 'EPISODIC', 'PROCEDURAL');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- CreateEnum WorkflowStatus
DO $$ BEGIN
    CREATE TYPE "WorkflowStatus" AS ENUM ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- AlterTable project_repositories (drop default if needed)
ALTER TABLE "project_repositories" ALTER COLUMN "updatedAt" DROP DEFAULT;

-- CreateTable conversation_threads
CREATE TABLE IF NOT EXISTS "conversation_threads" (
    "id" TEXT NOT NULL,
    "projectId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "title" TEXT,
    "cognitiveState" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conversation_threads_pkey" PRIMARY KEY ("id")
);

-- CreateTable chat_messages
CREATE TABLE IF NOT EXISTS "chat_messages" (
    "id" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "role" "MessageRole" NOT NULL,
    "content" TEXT NOT NULL,
    "cognitiveAction" TEXT,
    "styleModifier" TEXT,
    "reasoning" TEXT,
    "toolCalls" JSONB,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "chat_messages_pkey" PRIMARY KEY ("id")
);

-- CreateTable memory_records
CREATE TABLE IF NOT EXISTS "memory_records" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "projectId" TEXT,
    "type" "MemoryType" NOT NULL,
    "key" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "importance" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "accessCount" INTEGER NOT NULL DEFAULT 0,
    "lastAccessed" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "metadata" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_records_pkey" PRIMARY KEY ("id")
);

-- CreateTable agent_workflow_runs
CREATE TABLE IF NOT EXISTS "agent_workflow_runs" (
    "id" TEXT NOT NULL,
    "projectId" TEXT NOT NULL,
    "threadId" TEXT,
    "workflowType" TEXT NOT NULL,
    "status" "WorkflowStatus" NOT NULL DEFAULT 'PENDING',
    "inputPayload" JSONB,
    "result" JSONB,
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "agent_workflow_runs_pkey" PRIMARY KEY ("id")
);

-- CreateIndexes
CREATE INDEX IF NOT EXISTS "conversation_threads_projectId_userId_idx" ON "conversation_threads"("projectId", "userId");
CREATE INDEX IF NOT EXISTS "chat_messages_threadId_createdAt_idx" ON "chat_messages"("threadId", "createdAt" ASC);
CREATE INDEX IF NOT EXISTS "memory_records_userId_projectId_type_idx" ON "memory_records"("userId", "projectId", "type");
CREATE INDEX IF NOT EXISTS "memory_records_key_idx" ON "memory_records"("key");
CREATE INDEX IF NOT EXISTS "agent_workflow_runs_projectId_status_idx" ON "agent_workflow_runs"("projectId", "status");

-- Foreign Keys
DO $$ BEGIN
    ALTER TABLE "conversation_threads" ADD CONSTRAINT "conversation_threads_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "projects"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "conversation_threads" ADD CONSTRAINT "conversation_threads_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "chat_messages" ADD CONSTRAINT "chat_messages_threadId_fkey" FOREIGN KEY ("threadId") REFERENCES "conversation_threads"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "memory_records" ADD CONSTRAINT "memory_records_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "agent_workflow_runs" ADD CONSTRAINT "agent_workflow_runs_projectId_fkey" FOREIGN KEY ("projectId") REFERENCES "projects"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "agent_workflow_runs" ADD CONSTRAINT "agent_workflow_runs_threadId_fkey" FOREIGN KEY ("threadId") REFERENCES "conversation_threads"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Enable pgvector and configure semantic chat search (Tier 2 chat history traversal)
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE "chat_messages" ADD COLUMN IF NOT EXISTS "embedding" vector(768);
CREATE INDEX IF NOT EXISTS "chat_messages_embedding_hnsw_idx" ON "chat_messages" USING hnsw ("embedding" vector_cosine_ops);
