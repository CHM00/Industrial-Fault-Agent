import os
import json
import asyncio
import re
import threading
import time
from pathlib import Path
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from openai import AsyncOpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from Milvus import MilvusClass
from tavily import TavilyClient
from dotenv import load_dotenv
from Local_Model import Load_Model
from Knowledge_Grpah import KnowledgeGraph
from context_router import ContextRouter
from typing import AsyncGenerator, List, Dict, Tuple
from skills import SkillRegistry
from orchestrator import SkillOrchestrator
from intent_router_service import IntentRouterService
from langfuse_monitor import LangfuseMonitor
from adaptive_retriever import AdaptiveRetriever
from offline_cache import OfflineCachePool
# 加载环境变量(.env文件)
load_dotenv()


class SmartAgentBrain:
    def __init__(self, LOCAL_LLM=False):
        # ================= 配置区域 =================
        self.ARK_API_KEY = os.environ.get("ARK_API_KEY")
        self.ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.LLM_MODEL = "deepseek-ai/DeepSeek-V3"
        self.EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

        self.MILVUS_URI = os.environ.get("URL")
        self.MILVUS_TOKEN = os.environ.get("Token")

        self.search_web_key = os.environ.get("trivily_key")

        # 初始化 LLM 客户端
        self.aclient = AsyncOpenAI(
            api_key=self.ARK_API_KEY,
            base_url=self.ARK_BASE_URL,
        )

        # 检索工具
        self.client = TavilyClient(self.search_web_key)

        # === 上下文与压缩配置 ===
        self.history = []
        self.max_history_len = 20  # 放宽历史记录上限，交由动态预算管理器处理
        self.running_summary = ""  # 滚动摘要，存储被折叠的旧历史
        self.max_context_tokens = 1600  # 最大 Token 预算

        # 初始化类
        self.milvus = MilvusClass()
        self.milvus.connect_milvus()
        self.memory_collection = self.milvus.memory_collection
        self.collection = self.milvus.food_collection

        # 初始化图谱（连接失败时降级为 None）
        self.kg = KnowledgeGraph()
        try:
            self.kg.connect()
            self._kg_connected = True
        except Exception as e:
            print(f"[Brain] Neo4j 连接失败，图谱功能降级: {e}")
            self._kg_connected = False

        self.skill_registry = SkillRegistry(self.kg)
        self.skill_orchestrator = SkillOrchestrator(self.skill_registry)
        self.context_router = ContextRouter(embedding_func=self.milvus.embedding if hasattr(self.milvus, "embedding") else None)
        # 初始化本地类 (仅保留 ASR/CAM 等非 LLM 本地模型)
        self.local_model = Load_Model()
        self.LOCAL_LLM = LOCAL_LLM  # 强制使用 API 模型

        from intent_router_bert import BertIntentRouter

        cross_modal_checkpoint = "./BERT-Finetuing/cross_modal_output/best.pt"
        speaker_feature_dim = 192

        self.intent_router = BertIntentRouter(
            model_dir="./BERT-Finetuing/final_intent_model",
            cross_modal_checkpoint_path=cross_modal_checkpoint if Path(cross_modal_checkpoint).exists() else "",
            speaker_feature_dim=speaker_feature_dim,
        )
        self.intent_router_service = IntentRouterService(
            self.intent_router,
            llm_client=self.aclient,
            llm_model=self.LLM_MODEL,
            skill_registry=self.skill_registry,
            llm_enabled=True,
        )

        self.adaptive_retriever = AdaptiveRetriever(
            milvus_instance=self.milvus,
            kg_instance=self,
        )

        self.offline_cache = OfflineCachePool(cache_dir="./output/offline_cache")

        self._safety_threshold = 0.3

        self.alignment_trigger_count = 0
        self._alignment_interval = 20

        # 可观测性：未安装或未配置 Langfuse 时会自动降级
        self.monitor = LangfuseMonitor(service_name="sensevoice-agent")