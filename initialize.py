"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import constants as ct
import utils

############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################


def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE), when="D"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def initialize_retriever():
    """
    Retrieverを作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    loader = CSVLoader(ct.RAG_SOURCE_PATH)
    docs = loader.load()

    docs_all = []
    for doc in docs:
        docs_all.append(doc.page_content)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embedding=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    bm25_retriever = BM25Retriever.from_texts(
        docs_all, preprocess_func=utils.preprocess_func, k=ct.TOP_K
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=ct.RETRIEVER_WEIGHTS
    )

    st.session_state.retriever = ensemble_retriever
