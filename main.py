"""
æœ€é©åŒ–ã•ã‚ŒãŸFastAPI ã‚µãƒãƒ¼ãƒˆæ¤œç´¢ã‚·ã‚¹ãƒ†ãƒ  (Render.comå¯¾å¿œç‰ˆ)
- èµ·å‹•æ™‚é–“å‰Šæ¸› (lazy loading)
- FAISS ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æœ€é©åŒ–
- æ—¢å­˜ã®é™çš„ãƒ•ã‚¡ã‚¤ãƒ«æ§‹æˆã¨ã®äº’æ›æ€§ç¶­æŒ
"""

from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import auth_utils as au  # 管理画面の多要素認証・パスワードハッシュ等（auth_utils.py）
from contextlib import asynccontextmanager
from functools import lru_cache

from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import pathlib
import csv
import secrets
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import re
import numpy as np
from collections import Counter
import unicodedata

# ============================================
# è¨­å®š
# ============================================

APP_TITLE = "ã‚µãƒãƒ¼ãƒˆæ¤œç´¢ï¼ˆå‹•ç”» + FAQï¼‰æœ€é©åŒ–ç‰ˆ"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

DEFAULT_TOP_K = 10
DEFAULT_PAGE_LIMIT = 10
MAX_PAGE_LIMIT = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.3  # 類似度スコアのしきい値（0.0-1.0）
SEMANTIC_WEIGHT = 0.6  # セマンティック検索の重み（デフォルト: 60%）
TITLE_WEIGHT = 0.4     # タイトル一致の重み（デフォルト: 40%）

# FAQ検索: キーワード一致度がこの値以上であれば、セマンティックスコアが
# 閾値未満でも検索結果に含める（質問・回答・手順・キーワードへの直接一致を優先）
FAQ_KEYWORD_MATCH_MIN_SCORE = 0.3

# 動画transcriptのチャンク分割設定
# 使用モデル(paraphrase-multilingual-MiniLM-L12-v2)の最大シーケンス長は128トークン。
# 日本語は概ね1トークン=1〜1.5文字のため、安全マージンを見て1チャンクを180文字以内に収める。
# これによりtranscriptが長い動画でも全体を検索対象にできる（チャンク単位でベクトル化）。
VIDEO_CHUNK_MAX_CHARS = 180   # 1チャンクあたりの最大文字数
VIDEO_CHUNK_OVERLAP = 30      # チャンク間の重複文字数（文の境界をまたぐ内容の検索漏れを防ぐ）

# ç®¡ç†è€…èªè¨¼
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "abc123")

BASE_DIR = pathlib.Path(__file__).parent

# Persistent Disk対応: 環境変数 DATA_DIR が設定されている場合はそちらを使用
# Render.comでは DATA_DIR=/data を環境変数に設定する
_DATA_DIR_ENV = os.getenv("DATA_DIR", "")
DATA_DIR = pathlib.Path(_DATA_DIR_ENV) if _DATA_DIR_ENV else BASE_DIR

# データファイルパス（Disk上に保持）
DATA_PATH        = DATA_DIR / "data.json"
SYNONYMS_PATH    = DATA_DIR / "synonyms.json"
FAQ_PATH         = DATA_DIR / "faq.json"
CONFIG_PATH      = DATA_DIR / "config.json"
USERS_PATH       = DATA_DIR / "users.json"
SEARCH_LOG_PATH  = DATA_DIR / "search_logs.csv"
API_KEYS_PATH    = DATA_DIR / "api_keys.json"      # 外部システム連携用APIキー
FAQ_HISTORY_PATH = DATA_DIR / "faq_history.json"   # FAQ変更履歴

# アップロードファイルパス（Disk上に保持）
FILES_DIR        = DATA_DIR / "files"
FILES_MANUALS    = FILES_DIR / "manuals"
FILES_TOOLS      = FILES_DIR / "tools"
FILES_INSTALLERS = FILES_DIR / "installers"
FILES_OTHER      = FILES_DIR / "other"

# 動画embeddingキャッシュパス（Disk上に保持）
# data.jsonの内容が変わらない限り、起動のたびに重いembedding計算を
# 繰り返さずに済むよう、チャンクのベクトルをファイルとして保存しておく。
CACHE_DIR                  = DATA_DIR / "cache"
VIDEO_EMBEDDINGS_CACHE_PATH = CACHE_DIR / "video_embeddings.npy"
VIDEO_CHUNK_META_CACHE_PATH = CACHE_DIR / "video_chunks_meta.json"

# 静的ファイルパス（コードと同梱、デプロイ毎にリセットされる）
frontend_path = BASE_DIR / "frontend"
admin_path    = BASE_DIR / "admin_ui"

# ============================================
# ã‚°ãƒ­ãƒ¼ãƒãƒ«çŠ¶æ…‹ç®¡ç†
# ============================================

class AppState:
    """ã‚¢ãƒ—ãƒªã‚±ãƒ¼ã‚·ãƒ§ãƒ³çŠ¶æ…‹ã®ä¸€å…ƒç®¡ç†"""
    def __init__(self):
        # å‹•ç”»æ¤œç´¢ç”¨
        self.videos: List[Dict[str, Any]] = []
        self.text_corpus: List[str] = []
        self.synonyms: Dict[str, List[str]] = {}
        self.model: Optional[SentenceTransformer] = None
        self.video_index: Optional[faiss.Index] = None
        self.video_embeddings: Optional[np.ndarray] = None
        self.video_chunk_map: List[int] = []  # チャンクindex -> videosのインデックス（チャンク分割検索用）
        
        # FAQæ¤œç´¢ç”¨
        self.faq_data: Dict[str, Any] = {}
        self.faq_items_flat: List[Dict[str, Any]] = []
        self.faq_corpus: List[str] = []
        self.faq_index: Optional[faiss.Index] = None
        self.faq_embeddings: Optional[np.ndarray] = None
        
        # åˆæœŸåŒ–çŠ¶æ…‹
        self.video_loaded = False
        self.faq_loaded = False
        self.model_loaded = False
        
    async def ensure_model_loaded(self):
        """ãƒ¢ãƒ‡ãƒ«ã®é…å»¶ãƒ­ãƒ¼ãƒ‰"""
        if not self.model_loaded:
            print(f"ðŸ”„ Loading model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model_loaded = True
            print("âœ… Model loaded")
    
    async def ensure_video_loaded(self):
        """動画データの遅延ロード（embeddingキャッシュ対応）"""
        if not self.video_loaded:
            await self.ensure_model_loaded()
            print("🔄 Loading video data...")
            
            # データ読み込み
            if DATA_PATH.exists():
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    self.videos = json.load(f)
            
            if SYNONYMS_PATH.exists():
                with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            
            # data.jsonの内容ハッシュを計算（embeddingキャッシュの整合性チェック用）
            current_hash = _compute_file_hash(DATA_PATH)
            
            # ------------------------------------------------------------
            # embeddingキャッシュの読み込みを試みる。
            # data.jsonの内容とモデル名が前回キャッシュ作成時と一致していれば、
            # 重いembedding計算をスキップしてキャッシュから復元する。
            # ------------------------------------------------------------
            cache_loaded = False
            if VIDEO_EMBEDDINGS_CACHE_PATH.exists() and VIDEO_CHUNK_META_CACHE_PATH.exists():
                try:
                    with open(VIDEO_CHUNK_META_CACHE_PATH, "r", encoding="utf-8") as f:
                        cache_meta = json.load(f)
                    
                    hash_match  = cache_meta.get("data_hash") == current_hash
                    model_match = cache_meta.get("model") == EMBEDDING_MODEL
                    chunk_settings_match = (
                        cache_meta.get("chunk_max_chars") == VIDEO_CHUNK_MAX_CHARS and
                        cache_meta.get("chunk_overlap") == VIDEO_CHUNK_OVERLAP
                    )
                    
                    if hash_match and model_match and chunk_settings_match:
                        self.text_corpus = cache_meta["text_corpus"]
                        self.video_chunk_map = cache_meta["video_chunk_map"]
                        self.video_embeddings = np.load(VIDEO_EMBEDDINGS_CACHE_PATH)
                        cache_loaded = True
                        print(f"⚡ Video embeddings loaded from cache: {len(self.text_corpus)} chunks (skipped re-encoding)")
                    else:
                        reason = []
                        if not hash_match: reason.append("data.json changed")
                        if not model_match: reason.append("model changed")
                        if not chunk_settings_match: reason.append("chunk settings changed")
                        print(f"♻️ Video embeddings cache invalid ({', '.join(reason)}), recomputing...")
                except Exception as e:
                    print(f"⚠️ Failed to load video embeddings cache: {e}")
                    cache_loaded = False
            
            if not cache_loaded:
                # ------------------------------------------------------------
                # コーパス構築（チャンク分割対応）
                # transcriptが長い動画（モデルの最大トークン長128を超える場合）でも
                # 全体を検索対象にするため、タイトル+説明文+transcriptチャンクの単位で
                # 複数ベクトルを生成する。同じ動画の複数チャンクは video_chunk_map で
                # 動画インデックスに紐付けられ、検索時にスコアを集約する。
                # ------------------------------------------------------------
                self.text_corpus = []
                self.video_chunk_map = []

                for v_idx, v in enumerate(self.videos):
                    title = v.get("title", "") or ""
                    description = v.get("description", "") or ""
                    transcript = v.get("transcript", "") or ""
                    base = f"{title} {description}".strip()

                    if transcript:
                        text_chunks = chunk_text(transcript)
                        if not text_chunks:
                            text_chunks = [""]
                        for tc in text_chunks:
                            combined = f"{base} {tc}".strip()
                            self.text_corpus.append(normalize_text(combined))
                            self.video_chunk_map.append(v_idx)
                    else:
                        # transcriptが無い動画はタイトル+説明文のみで1チャンク
                        self.text_corpus.append(normalize_text(base))
                        self.video_chunk_map.append(v_idx)

                # embedding計算（ここが最も重い処理）
                if self.text_corpus:
                    self.video_embeddings = self.model.encode(
                        self.text_corpus, 
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=64
                    )
                    faiss.normalize_L2(self.video_embeddings)
                    
                    # 次回起動時に再利用できるようDiskにキャッシュ保存
                    try:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        np.save(VIDEO_EMBEDDINGS_CACHE_PATH, self.video_embeddings)
                        with open(VIDEO_CHUNK_META_CACHE_PATH, "w", encoding="utf-8") as f:
                            json.dump({
                                "data_hash": current_hash,
                                "model": EMBEDDING_MODEL,
                                "chunk_max_chars": VIDEO_CHUNK_MAX_CHARS,
                                "chunk_overlap": VIDEO_CHUNK_OVERLAP,
                                "text_corpus": self.text_corpus,
                                "video_chunk_map": self.video_chunk_map,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                            }, f, ensure_ascii=False)
                        print(f"💾 Video embeddings cache saved: {len(self.text_corpus)} chunks")
                    except Exception as e:
                        print(f"⚠️ Failed to save video embeddings cache: {e}")
            
            # FAISS インデックス構築（キャッシュ読み込み時・新規計算時ともに毎回実行。
            # インデックス構築自体はO(n)程度で高速なため、キャッシュ対象には含めない）
            if self.video_embeddings is not None and len(self.video_embeddings) > 0:
                self.video_index = build_optimized_index(self.video_embeddings)
            
            self.video_loaded = True
            print(f"✅ Video data loaded: {len(self.videos)} videos, {len(self.text_corpus)} chunks")
    
    async def ensure_faq_loaded(self):
        """FAQãƒ‡ãƒ¼ã‚¿ã®é…å»¶ãƒ­ãƒ¼ãƒ‰"""
        if not self.faq_loaded:
            await self.ensure_model_loaded()
            print("ðŸ”„ Loading FAQ data...")
            
            if FAQ_PATH.exists():
                print(f"✅ FAQ file found: {FAQ_PATH}")
                try:
                    # UTF-8で読み込み試行
                    with open(FAQ_PATH, "r", encoding="utf-8") as f:
                        self.faq_data = json.load(f)
                    print(f"✅ FAQ file loaded (UTF-8), keys: {list(self.faq_data.keys())}")
                except UnicodeDecodeError:
                    # UTF-8で失敗した場合、Shift-JIS (cp932) を試行
                    print(f"⚠️ UTF-8 decode failed, trying cp932...")
                    try:
                        with open(FAQ_PATH, "r", encoding="cp932") as f:
                            self.faq_data = json.load(f)
                        print(f"✅ FAQ file loaded (cp932), keys: {list(self.faq_data.keys())}")
                    except Exception as e:
                        print(f"❌ Failed to load FAQ file with cp932: {e}")
                        self.faq_data = {}
                except Exception as e:
                    print(f"❌ Failed to load FAQ file: {e}")
                    self.faq_data = {}
            else:
                print(f"❌ FAQ file not found: {FAQ_PATH}")
                print(f"   BASE_DIR: {BASE_DIR}")
                self.faq_data = {}
            
            # FAQ ã‚¢ã‚¤ãƒ†ãƒ ã‚’ãƒ•ãƒ©ãƒƒãƒˆåŒ–
            # å¯¾å¿œãƒ•ã‚©ãƒ¼ãƒžãƒƒãƒˆ:
            #   A) {"faqs": [...], "meta": {...}}  â† faqsé…åˆ—ç›´æŽ¥
            #   B) {"ã‚«ãƒ†ã‚´ãƒªå": [...], ...}       â† ã‚«ãƒ†ã‚´ãƒªè¾žæ›¸
            self.faq_items_flat = []

            # ãƒ•ã‚©ãƒ¼ãƒžãƒƒãƒˆA: "faqs" ã‚­ãƒ¼ã«é…åˆ—ãŒå…¥ã£ã¦ã„ã‚‹å ´åˆ
            if "faqs" in self.faq_data and isinstance(self.faq_data["faqs"], list):
                print(f"📋 Processing {len(self.faq_data['faqs'])} FAQ items from 'faqs' array")
                for item in self.faq_data["faqs"]:
                    if isinstance(item, dict):
                        # フィールド正規化: faq_id → id, answer_steps → steps
                        normalized_item = item.copy()
                        if "faq_id" in normalized_item and "id" not in normalized_item:
                            normalized_item["id"] = normalized_item.pop("faq_id")
                        if "answer_steps" in normalized_item and "steps" not in normalized_item:
                            normalized_item["steps"] = normalized_item.pop("answer_steps")
                        
                        # 不要なフィールドを削除
                        for field in ["manual_ref", "confidence", "support_based"]:
                            if field in normalized_item:
                                del normalized_item[field]
                        
                        # utterances がない場合は question で代用
                        if "utterances" not in normalized_item and "question" in normalized_item:
                            normalized_item["utterances"] = [normalized_item["question"]]
                        self.faq_items_flat.append(normalized_item)
                print(f"✅ Normalized {len(self.faq_items_flat)} FAQ items")
            else:
                # ãƒ•ã‚©ãƒ¼ãƒžãƒƒãƒˆB: ã‚«ãƒ†ã‚´ãƒªè¾žæ›¸å½¢å¼
                for category_key, items in self.faq_data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                if "category" not in item:
                                    item["category"] = category_key
                                self.faq_items_flat.append(item)
            
            # コーパス構築（question / utterances / steps / answer / keywords / note を統合）
            # FAQ items合計のログ出力
            print(f"📊 Total FAQ items loaded: {len(self.faq_items_flat)}")
            
            self.faq_corpus = []
            for item in self.faq_items_flat:
                text_parts = [
                    item.get("question", ""),
                    " ".join(item.get("utterances", [])),
                    " ".join(item.get("steps", [])),
                    item.get("answer", ""),          # steps正規化されなかった場合の回答本文も対象に
                    " ".join(item.get("keywords", [])),
                    " ".join(item.get("tags", [])),  # tags も検索対象に
                    item.get("note", ""),            # ヒント・補足情報も検索対象に
                    item.get("intent", ""),
                    item.get("category", ""),
                ]
                combined = " ".join(filter(None, text_parts))
                self.faq_corpus.append(normalize_text(combined))
            
            # FAISS インデックス構築
            if self.faq_corpus:
                self.faq_embeddings = self.model.encode(
                    self.faq_corpus,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                faiss.normalize_L2(self.faq_embeddings)
                self.faq_index = build_optimized_index(self.faq_embeddings)
            
            self.faq_loaded = True
            print(f"âœ… FAQ data loaded: {len(self.faq_items_flat)} items")

state = AppState()

# ============================================
# ãƒ¦ãƒ¼ãƒ†ã‚£ãƒªãƒ†ã‚£é–¢æ•°
# ============================================

@lru_cache(maxsize=1000)
def normalize_text(text: str) -> str:
    """ãƒ†ã‚­ã‚¹ãƒˆæ­£è¦åŒ–ï¼ˆã‚­ãƒ£ãƒƒã‚·ãƒ¥ä»˜ãï¼‰"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text: str, max_chars: int = VIDEO_CHUNK_MAX_CHARS, overlap: int = VIDEO_CHUNK_OVERLAP) -> List[str]:
    """
    長いテキスト（動画のtranscript等）を、モデルの最大トークン長に収まる
    サイズのチャンクに分割する。

    句点（。！？）を優先した自然な区切りで分割し、1文がmax_charsを超える
    場合は文字数で強制分割する。チャンク間にoverlap文字分の重複を持たせ、
    文の境界をまたぐ内容が検索から漏れるのを防ぐ。

    Args:
        text: 分割対象のテキスト
        max_chars: 1チャンクあたりの最大文字数
        overlap: チャンク間の重複文字数

    Returns:
        チャンク文字列のリスト（textが空ならば空リスト）
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 句点等で分割（区切り文字自体は各文の末尾に残す）
    sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]

    chunks = []
    current = ""

    for sent in sentences:
        # 1文自体がmax_charsを超える場合は文字数で強制分割
        while len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = current[-overlap:] if len(current) > overlap else ""
            chunks.append(sent[:max_chars])
            sent = sent[max(max_chars - overlap, 1):]

        if len(current) + len(sent) <= max_chars:
            current += sent
        else:
            if current:
                chunks.append(current)
                current = current[-overlap:] if len(current) > overlap else ""
            current += sent

    if current.strip():
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]

def _compute_file_hash(file_path: pathlib.Path) -> str:
    """
    ファイルの内容からSHA-256ハッシュ値を計算する。
    embeddingキャッシュがdata.jsonの内容と一致しているかを判定するために使う。
    """
    if not file_path.exists():
        return ""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def expand_with_synonyms(query: str, synonyms: Dict[str, Any], language: str = "ja") -> str:
    """
    同義語展開（言語別対応）
    
    Args:
        query: 検索クエリ
        synonyms: 言語別同義語辞書 {"ja": {...}, "en": {...}}
        language: 言語コード（"ja", "en"）
    
    Returns:
        展開されたクエリ
    """
    expanded_terms = [query]
    
    # 言語別の同義語辞書を取得
    lang_synonyms = synonyms.get(language, {}) if isinstance(synonyms, dict) else {}
    
    # 同義語展開
    for term, syns in lang_synonyms.items():
        if term.lower() in query.lower():
            expanded_terms.extend(syns)
    
    return " ".join(expanded_terms)

def _char_bigrams(text: str) -> list:
    """文字bi-gram（2文字の連続組）のリストを生成する。
    日本語は分かち書きされないため、単語分割の代わりに
    文字N-gramで部分一致度を評価するのに用いる。"""
    if len(text) < 2:
        return [text] if text else []
    return [text[i:i+2] for i in range(len(text) - 1)]


def title_match_score(query: str, title: str) -> float:
    """
    タイトルへのキーワード一致度を計算（0.0～1.0）

    日本語クエリはスペース区切りなしで入力されることが多いため、
    単語分割（str.split）だけでは複合語（例:「パスワード変更方法」）が
    一致していても正しく評価できない。
    そのため、以下の2種類のスコアのうち高い方を採用する:
      1) スペース区切りの単語一致率（分かち書き入力・英語クエリ向け）
      2) 文字bi-gramの一致率（日本語の複合語向け）

    Args:
        query: 検索クエリ
        title: 動画タイトル / FAQ質問文

    Returns:
        一致度スコア（0.0～1.0）
    """
    if not query or not title:
        return 0.0

    query_norm = normalize_text(query)
    title_norm = normalize_text(title)

    if not query_norm:
        return 0.0

    # クエリ全体がタイトルにそのまま含まれる場合は満点
    if query_norm in title_norm:
        return 1.0

    # 1) スペース区切りの単語一致率
    query_words = query_norm.split()
    if len(query_words) > 1:
        matched_words = sum(1 for w in query_words if w and w in title_norm)
        word_score = matched_words / len(query_words)
    else:
        word_score = 0.0

    # 2) 文字bi-gramの一致率（日本語の複合語に対応）
    bigrams = _char_bigrams(query_norm.replace(' ', ''))
    if bigrams:
        matched_bigrams = sum(1 for bg in bigrams if bg in title_norm)
        bigram_score = matched_bigrams / len(bigrams)
    else:
        bigram_score = 0.0

    return max(word_score, bigram_score)
def faq_keyword_match_score(query: str, item: Dict[str, Any]) -> float:
    """
    FAQ項目全体（質問・回答・手順・キーワード・タグ・補足）に対する
    クエリの一致度を計算する。

    embeddingによる意味検索だけでは、専門用語やキーワードの
    完全一致・部分一致を拾いきれないことがあるため、
    質問文だけでなく回答・手順・キーワード等も含めた全文をtitle_match_scoreと
    同じロジック（単語一致 / 文字bi-gram一致の複合判定）で評価する。

    Args:
        query: 検索クエリ
        item: FAQ項目（question/answer/steps/keywords/tags/noteを含む辞書）

    Returns:
        一致度スコア（0.0～1.0）
    """
    search_text_parts = [
        item.get("question", "") or "",
        item.get("answer", "") or "",
        " ".join(item.get("steps", []) or []),
        " ".join(item.get("utterances", []) or []),
        " ".join(item.get("keywords", []) or []),
        " ".join(item.get("tags", []) or []),
        item.get("note", "") or "",
    ]
    combined_text = " ".join(filter(None, search_text_parts))
    return title_match_score(query, combined_text)




def build_optimized_index(embeddings: np.ndarray) -> faiss.Index:
    """æœ€é©åŒ–ã•ã‚ŒãŸFAISSã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æ§‹ç¯‰"""
    n, dim = embeddings.shape
    
    if n <= 1000:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        nlist = min(100, n // 10)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 10
    
    return index

def parse_logs() -> List[Dict[str, Any]]:
    """
    ログパース（管理画面用）
    CSVフォーマット: timestamp, type, query, result_id
    ヘッダー行は自動スキップ
    """
    if not SEARCH_LOG_PATH.exists():
        print(f"⚠️ Log file not found: {SEARCH_LOG_PATH}")
        return []

    rows = []
    skipped = 0
    try:
        with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # ヘッダー行スキップ
                if i == 0 and len(row) >= 1 and row[0].strip().lower() == "timestamp":
                    continue
                if len(row) < 1:
                    continue
                try:
                    timestamp_str = row[0].strip()
                    result_type   = row[1].strip() if len(row) > 1 else ""
                    query         = row[2].strip() if len(row) > 2 else ""
                    result_id     = row[3].strip() if len(row) > 3 else ""

                    if not timestamp_str:
                        skipped += 1
                        continue

                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    rows.append({
                        "dt":          dt,
                        "result_type": result_type,
                        "query":       query,
                        "result_id":   result_id,
                    })
                except Exception:
                    skipped += 1
    except Exception as e:
        print(f"❌ Failed to parse logs: {e}")
        return []

    print(f"📊 Logs parsed: {len(rows)} rows, {skipped} skipped")
    return rows


# ============================================
# èªè¨¼
# ============================================

def _client_ip(request: Request) -> str:
    return (request.client.host if request.client else "unknown")

def verify_admin(authorization: Optional[str] = Header(None)) -> str:
    """
    管理者認証（ログイン済みトークン方式）。
    ログイン（パスワード＋認証アプリのコード）に成功すると発行されるトークンを
    「Authorization: Bearer <token>」で受け取り、ユーザー名を返す。
    パスワード変更・MFAリセットが行われたユーザーの古いトークンは無効になる（token_epoch照合）。
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login required",
                            headers={"WWW-Authenticate": "Bearer"})
    payload = au.parse_token(authorization[7:].strip(), "session")
    if not payload:
        raise HTTPException(status_code=401, detail="Session expired or invalid",
                            headers={"WWW-Authenticate": "Bearer"})
    user = au.find_user(au.load_users(), payload.get("u", ""))
    if not user or user.get("token_epoch", 1) != payload.get("e"):
        raise HTTPException(status_code=401, detail="Session expired or invalid",
                            headers={"WWW-Authenticate": "Bearer"})
    return user["username"]

# ============================================
# 外部システム連携API - 認証（APIキー方式）
# ============================================
# 管理画面はBasic認証(verify_admin)のまま、
# 外部システムからのアクセスはAPIキー(X-API-Keyヘッダー)で認証する。
# api_keys.json 形式:
#   {"keys": [{"key": "...", "name": "システム名", "enabled": true, "created_at": "..."}]}

def load_api_keys() -> list:
    """api_keys.jsonからキー一覧を読み込む。失敗時は空リスト。"""
    try:
        if not API_KEYS_PATH.exists():
            return []
        with open(API_KEYS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("keys", [])
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return []

def save_api_keys(keys: list):
    with open(API_KEYS_PATH, "w", encoding="utf-8") as f:
        json.dump({"keys": keys}, f, ensure_ascii=False, indent=2)

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    外部システム向けAPI(/api/v1/*)の認証。
    有効なAPIキーが一致した場合、そのキーのname（呼び出し元システム名）を返す。
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is required")

    for entry in load_api_keys():
        if entry.get("key") == x_api_key:
            if not entry.get("enabled", True):
                raise HTTPException(status_code=403, detail="This API key has been disabled")
            return entry.get("name", "unknown")

    raise HTTPException(status_code=401, detail="Invalid API key")

# ============================================
# 外部システム連携API - FAQ変更履歴
# ============================================

def record_faq_history(faq_id: str, action: str, changed_by: str, before, after):
    """
    FAQの作成・更新・削除を履歴として記録する。
    更新時は差分がある項目のみを保存する。
    """
    try:
        if FAQ_HISTORY_PATH.exists():
            with open(FAQ_HISTORY_PATH, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        else:
            history_data = {"history": []}

        changes = {}
        if action == "update" and before and after:
            keys = set(before.keys()) | set(after.keys())
            for k in keys:
                old_v, new_v = before.get(k), after.get(k)
                if old_v != new_v:
                    changes[k] = {"before": old_v, "after": new_v}
        elif action == "create":
            changes = after or {}
        elif action == "delete":
            changes = before or {}

        entry = {
            "faq_id": faq_id,
            "action": action,           # create / update / delete
            "changed_by": changed_by,   # admin または APIキーのname
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes": changes,
        }
        history_data.setdefault("history", []).append(entry)

        with open(FAQ_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to record FAQ history: {e}")

def get_faq_history(faq_id: str = None) -> list:
    """FAQ履歴を取得する。faq_id指定時はそのFAQのみに絞り込む。"""
    try:
        if not FAQ_HISTORY_PATH.exists():
            return []
        with open(FAQ_HISTORY_PATH, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        rows = history_data.get("history", [])
        if faq_id:
            rows = [r for r in rows if r.get("faq_id") == faq_id]
        return rows
    except Exception as e:
        print(f"❌ Failed to load faq_history.json: {e}")
        return []


# ============================================
# Lifespanç®¡ç†
# ============================================

# ============================================
# 起動時の初期化処理
# ============================================

def initialize_files():
    """必要なファイルを初期化（Persistent Disk対応）"""
    try:
        # Persistent Disk上のディレクトリを作成
        for d in [DATA_DIR, FILES_DIR, FILES_MANUALS, FILES_TOOLS, FILES_INSTALLERS, FILES_OTHER, CACHE_DIR]:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {d}")
        
        print(f"📂 Data directory: {DATA_DIR}")
        print(f"📂 Files directory: {FILES_DIR}")

        # ------------------------------------------------------------
        # Persistent Disk導入時の既存データ移行
        # DATA_DIR がソースコードと異なる場合（Disk使用時）、
        # ソースコードに同梱されている既存のデータファイルが存在し、
        # かつ Disk側にまだファイルが無い場合は、初回のみコピーする。
        # これにより、Disk導入前に蓄積していたデータが失われない。
        # ------------------------------------------------------------
        if DATA_DIR != BASE_DIR:
            migration_targets = [
                ("data.json",      BASE_DIR / "data.json",      DATA_PATH),
                ("faq.json",       BASE_DIR / "faq.json",       FAQ_PATH),
                ("synonyms.json",  BASE_DIR / "synonyms.json",  SYNONYMS_PATH),
                ("config.json",    BASE_DIR / "config.json",    CONFIG_PATH),
                ("users.json",     BASE_DIR / "users.json",     USERS_PATH),
                ("search_logs.csv", BASE_DIR / "search_logs.csv", SEARCH_LOG_PATH),
            ]
            for label, src, dst in migration_targets:
                try:
                    if src.exists() and src.is_file():
                        # Disk側が未作成、または空/デフォルトのままの場合のみコピー
                        should_copy = False
                        if not dst.exists():
                            should_copy = True
                        else:
                            # Disk側が極端に小さい（空配列やデフォルトのみ）場合も移行対象とする
                            try:
                                if dst.stat().st_size <= 4 and src.stat().st_size > 4:
                                    should_copy = True
                            except Exception:
                                pass

                        if should_copy:
                            import shutil
                            shutil.copy2(src, dst)
                            print(f"📦 Migrated existing {label} from source to Disk ({src} → {dst})")
                except Exception as e:
                    print(f"⚠️ Migration check failed for {label}: {e}")

        # config.json の初期化
        if not CONFIG_PATH.exists():
            print("📁 Creating config.json...")
            default_config = {
                "faq_search_enabled": True,
                "similarity_threshold": 0.3,
                "semantic_weight": 0.6,
                "title_weight": 0.4,
                "language_switcher_enabled": False
            }
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print("✅ config.json created")
        
        # data.json の初期化
        if not DATA_PATH.exists():
            print("📁 Creating data.json...")
            with open(DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print("✅ data.json created")
        
        # synonyms.json の初期化（言語別構造）
        if not SYNONYMS_PATH.exists():
            print("📁 Creating synonyms.json...")
            default_synonyms = {
                "ja": {
                    "プロッタ": ["プロッター", "大判プリンタ"],
                    "パスワード": ["PW", "pass"]
                },
                "en": {
                    "plotter": ["large format printer"],
                    "password": ["pw", "pass"]
                }
            }
            with open(SYNONYMS_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_synonyms, f, ensure_ascii=False, indent=2)
            print("✅ synonyms.json created")
        
        # faq.json の初期化
        if not FAQ_PATH.exists():
            print("📁 Creating faq.json...")
            default_faq = {
                "meta": {"version": "1.0", "count": 0},
                "faqs": []
            }
            with open(FAQ_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_faq, f, ensure_ascii=False, indent=2)
            print("✅ faq.json created")
        
        # users.json の初期化
        # 新規環境のみ初期管理者(admin/admin)を作成する。
        # 初回ログイン時に多要素認証の登録とパスワード変更が必須となる。
        if not USERS_PATH.exists():
            print("📁 Creating users.json...")
            au.save_users({"users": [au.new_user_record("admin", "admin", must_change_password=True)]})
            print("✅ users.json created")
        # 旧形式（平文パスワード・秘密の質問）を新形式へ移行
        au.migrate_users_file()

        # api_keys.json の初期化（外部システム連携用）
        if not API_KEYS_PATH.exists():
            print("📁 Creating api_keys.json...")
            default_api_keys = {"keys": []}
            with open(API_KEYS_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_api_keys, f, ensure_ascii=False, indent=2)
            print("✅ api_keys.json created")

        # faq_history.json の初期化（FAQ変更履歴）
        if not FAQ_HISTORY_PATH.exists():
            print("📁 Creating faq_history.json...")
            default_faq_history = {"history": []}
            with open(FAQ_HISTORY_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_faq_history, f, ensure_ascii=False, indent=2)
            print("✅ faq_history.json created")

        print("✅ File initialization completed")
        
    except Exception as e:
        print(f"❌ File initialization error: {e}")
        import traceback
        traceback.print_exc()

# 起動時に初期化を実行
initialize_files()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """èµ·å‹•æ™‚ã¯æœ€å°é™ã®åˆæœŸåŒ–ã®ã¿"""
    print("ðŸš€ Application starting...")
    
    # ãƒ­ã‚°ãƒ•ã‚¡ã‚¤ãƒ«åˆæœŸåŒ–
    if not SEARCH_LOG_PATH.exists():
        with open(SEARCH_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "type", "query", "result_id"])
    
    # ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªä½œæˆ
    admin_path.mkdir(parents=True, exist_ok=True)
    
    print("âœ… Application ready (lazy loading enabled)")
    yield
    print("ðŸ›‘ Application shutting down...")

# ============================================
# FastAPI ã‚¢ãƒ—ãƒªã‚±ãƒ¼ã‚·ãƒ§ãƒ³
# ============================================

app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ãƒ˜ãƒ«ã‚¹ãƒã‚§ãƒƒã‚¯
# ============================================

@app.get("/health")
async def health_check():
    """ãƒ˜ãƒ«ã‚¹ãƒã‚§ãƒƒã‚¯ï¼ˆRender.comç”¨ï¼‰"""
    return {
        "status": "healthy",
        "model_loaded": state.model_loaded,
        "video_loaded": state.video_loaded,
        "faq_loaded": state.faq_loaded,
        "faq_items_count": len(state.faq_items_flat),
        "faq_corpus_count": len(state.faq_corpus),
        "faq_index_available": state.faq_index is not None,
        "video_items_count": len(state.video_items_flat),
    }

# ============================================
# å‹•ç”»æ¤œç´¢ã‚¨ãƒ³ãƒ‰ãƒã‚¤ãƒ³ãƒˆ
# ============================================

@app.get("/search")
async def search_videos(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0),
    language: str = Query(None)
):
    """å‹•ç”»æ¤œç´¢ï¼ˆãƒšãƒ¼ã‚¸ãƒ³ã‚°å¯¾å¿œï¼‰"""
    await state.ensure_video_loaded()
    
    if not state.video_index:
        return {"items": [], "has_more": False, "total_visible": 0}
    
    normalized_query = normalize_text(query)
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
    
    # 設定から閾値を取得
    config = await get_config()
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    print(f"🔍 Video search: query='{query}', threshold={threshold}, config={config}")
    
    query_embedding = state.model.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # チャンク単位で検索するため、動画数ではなく総チャンク数を基準に取得件数を決定。
    # 同一動画の複数チャンクがヒットしても動画単位に集約するため、多めに取得する。
    total_chunks = len(state.video_chunk_map)
    k = min(max((offset + limit + 100) * 10, 500), total_chunks)
    distances, indices = state.video_index.search(query_embedding, k)
    
    # 設定から重み配分を取得
    semantic_weight = config.get("semantic_weight", SEMANTIC_WEIGHT)
    title_weight = config.get("title_weight", TITLE_WEIGHT)
    
    # チャンク単位のスコアを動画単位に集約する。
    # 同じ動画から複数のチャンクがヒットした場合は、最もスコアの高い
    # チャンクのスコアをその動画の意味検索スコアとして採用する。
    # これによりtranscriptの後半・中盤にマッチする内容でも動画を検索できる。
    best_score_per_video = {}   # video_idx -> semantic_score（最高値）
    all_scores = []  # 全チャンクスコアを記録（デバッグ用）

    for chunk_idx, score in zip(indices[0], distances[0]):
        if 0 <= chunk_idx < len(state.video_chunk_map):
            all_scores.append(float(score))
            v_idx = state.video_chunk_map[chunk_idx]
            s = float(score)
            if v_idx not in best_score_per_video or s > best_score_per_video[v_idx]:
                best_score_per_video[v_idx] = s

    results = []
    for v_idx, semantic_score in best_score_per_video.items():
        if not (0 <= v_idx < len(state.videos)):
            continue
        video = state.videos[v_idx].copy()

        # 言語フィルタリング
        video_lang = video.get("language", "ja")  # デフォルトは日本語
        if language != "all" and video_lang != language:
            continue

        # ハイブリッドスコアリング
        title_bonus = title_match_score(query, video.get("title", ""))
        hybrid_score = semantic_score * semantic_weight + title_bonus * title_weight

        video["score"] = hybrid_score
        video["semantic_score"] = semantic_score
        video["title_score"] = title_bonus

        # 閾値チェック（セマンティックスコアで判定）
        if semantic_score >= threshold:
            results.append(video)
    
    # デバッグ: 全体のスコア分布を表示
    if all_scores:
        print(f"  Total candidate chunks: {len(all_scores)} (from {total_chunks} total chunks)")
        print(f"  Unique videos matched: {len(best_score_per_video)}")
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Top 5 scores: {[f'{s:.3f}' for s in sorted(all_scores, reverse=True)[:5]]}")
        print(f"  Passed threshold ({threshold}): {len(results)} items")
    
    # タイトル完全一致を最優先、次点でハイブリッドスコアでソート
    query_norm = normalize_text(query)
    results.sort(key=lambda v: (
        -int(query_norm in normalize_text(v.get("title", ""))),  # タイトル一致を最優先
        -v["score"]  # 次点でハイブリッドスコア
    ))
    
    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total
    
    # デバッグ: スコアの範囲を確認
    if results:
        scores = [r["score"] for r in results[:10]]  # 上位10件のスコア
        print(f"🎬 Video search results: query='{query}', total={total}, items={len(items)}, threshold={threshold}")
        print(f"   Top 10 scores: {scores}")
    else:
        print(f"🎬 Video search results: query='{query}', total=0, items=0, threshold={threshold}")
    
    if paged:
        return {
            "items": items,
            "has_more": has_more,
            "total_visible": total,
            "offset": offset,
            "limit": limit
        }
    
    return {"items": items}

# ============================================
# FAQæ¤œç´¢ã‚¨ãƒ³ãƒ‰ãƒã‚¤ãƒ³ãƒˆ
# ============================================

@app.get("/faq/search")
async def search_faq(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0),
    language: str = Query(None)
):
    """FAQæ¤œç´¢ï¼ˆãƒšãƒ¼ã‚¸ãƒ³ã‚°å¯¾å¿œï¼‰"""
    await state.ensure_faq_loaded()
    
    if not state.faq_index:
        # FAISSインデックスが使用できない場合のフォールバック検索（簡易テキストマッチング）
        print(f"⚠️ FAQ index not available, using fallback text search")
        print(f"   FAQ items available: {len(state.faq_items_flat)}")
        print(f"   Query: '{query}'")
        
        normalized_query = normalize_text(query)
        expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
        print(f"   Normalized/expanded query: '{expanded_query}'")
        
        # クエリを単語に分割（分かち書き入力・英語クエリ向け）
        query_words = expanded_query.lower().split()
        # クエリの文字bi-gram（日本語の複合語向け。スペースは除去して連結evaluate）
        query_bigrams = _char_bigrams(expanded_query.lower().replace(' ', ''))
        print(f"   Query words: {query_words}, bigrams: {len(query_bigrams)}")
        
        # 各FAQアイテムとのマッチングスコアを計算
        scored_items = []
        for item in state.faq_items_flat:
            # 検索対象テキストを構築
            search_text = " ".join([
                item.get("question", "") or "",
                item.get("answer", "") or "",
                " ".join(item.get("utterances", []) or []),
                " ".join(item.get("steps", []) or []),
                " ".join(item.get("keywords", []) or []),
                " ".join(item.get("tags", []) or []),
                item.get("note", "") or "",
                item.get("category", "") or "",
            ]).lower()
            
            # マッチングスコアを計算
            # 1) 単語一致数（スペース区切り入力向け）
            word_matches = sum(1 for word in query_words if word and word in search_text)
            # 2) bi-gram一致率（日本語の複合語向け、0〜1に正規化してから重み付け）
            bigram_ratio = 0.0
            if query_bigrams:
                bigram_matches = sum(1 for bg in query_bigrams if bg in search_text)
                bigram_ratio = bigram_matches / len(query_bigrams)
            # bi-gram一致率をおおよそ単語一致と同程度のスケールに合わせて加算
            score = word_matches + bigram_ratio * max(len(query_words), 1)
            
            if score > 0:
                item_copy = item.copy()
                item_copy["score"] = float(score)
                scored_items.append(item_copy)
        
        print(f"   Matched items: {len(scored_items)}")
        
        # スコア順にソート（降順）
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        total = len(scored_items)
        items = scored_items[offset:offset + limit]
        has_more = (offset + limit) < total
        
        if paged:
            return {
                "items": items,
                "has_more": has_more,
                "total_visible": total,
                "offset": offset,
                "limit": limit,
                "fallback": True
            }
        
        return {"items": items, "fallback": True}
    
    # 設定から閾値を取得
    config = await get_config()
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    print(f"🔍 FAQ search: query='{query}', threshold={threshold}, config={config}")
    
    normalized_query = normalize_text(query)
    # 同義語展開で検索精度向上（Bug#5対応）
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
    query_embedding = state.model.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # 最大取得件数を100件に拡大
    k = min(offset + limit + 100, len(state.faq_items_flat))
    distances, indices = state.faq_index.search(query_embedding, k)
    
    # 設定から重み配分を取得
    semantic_weight = config.get("semantic_weight", SEMANTIC_WEIGHT)
    title_weight = config.get("title_weight", TITLE_WEIGHT)
    
    results = []
    all_scores = []  # 全スコアを記録
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.faq_items_flat):
            all_scores.append(float(score))
            item = state.faq_items_flat[idx].copy()
            
            # 言語フィルタリング
            item_lang = item.get("language", "ja")  # デフォルトは日本語
            if language != "all" and item_lang != language:
                continue
            
            # ハイブリッドスコアリング
            # 質問文だけでなく回答・手順・キーワード・タグ・補足も含めた
            # 全文一致度（keyword_bonus）を計算し、embeddingが拾いきれない
            # 専門用語・キーワードの完全一致検索を補完する。
            semantic_score = float(score)
            keyword_bonus = faq_keyword_match_score(query, item)
            hybrid_score = semantic_score * semantic_weight + keyword_bonus * title_weight
            
            item["score"] = hybrid_score
            item["semantic_score"] = semantic_score
            item["title_score"] = keyword_bonus
            
            # 閾値チェック: セマンティックスコアが閾値以上、
            # またはキーワード一致度が高い場合（質問・回答・手順・キーワードへの
            # 直接一致）は意味検索のスコアが低くても結果に含める
            if semantic_score >= threshold or keyword_bonus >= FAQ_KEYWORD_MATCH_MIN_SCORE:
                results.append(item)
    
    # デバッグ: 全体のスコア分布を表示
    if all_scores:
        print(f"  Total candidates: {len(all_scores)}")
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Top 5 scores: {[f'{s:.3f}' for s in sorted(all_scores, reverse=True)[:5]]}")
        print(f"  Passed threshold ({threshold}): {len(results)} items")
    
    # 質問文完全一致を最優先、次点でハイブリッドスコアでソート
    query_norm = normalize_text(query)
    results.sort(key=lambda v: (
        -int(query_norm in normalize_text(v.get("question", ""))),  # 質問文一致を最優先
        -v["score"]  # 次点でハイブリッドスコア
    ))
    
    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total
    
    # デバッグ: スコアの範囲を確認
    if results:
        scores = [r["score"] for r in results[:10]]  # 上位10件のスコア
        print(f"📋 FAQ search results: query='{query}', total={total}, items={len(items)}, threshold={threshold}")
        print(f"   Top 10 scores: {scores}")
    else:
        print(f"📋 FAQ search results: query='{query}', total=0, items=0, threshold={threshold}")
    
    if paged:
        return {
            "items": items,
            "has_more": has_more,
            "total_visible": total,
            "offset": offset,
            "limit": limit
        }
    
    return {"items": items}

# ============================================
# ç®¡ç†API - ãƒ‡ãƒ¼ã‚¿ç·¨é›†
# ============================================

@app.get("/admin/api/synonyms", dependencies=[Depends(verify_admin)])
async def get_synonyms():
    """åŒç¾©èªžè¾žæ›¸å–å¾—"""
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.put("/admin/api/synonyms", dependencies=[Depends(verify_admin)])
async def update_synonyms(data: dict):
    """同義語辞書更新"""
    print(f"💾 Saving synonyms: {len(data)} terms")
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Synonyms saved successfully")
    return {"status": "ok", "count": len(data)}

async def reload_video_data():
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ã®ãƒªãƒ­ãƒ¼ãƒ‰"""
    state.video_loaded = False
    await state.ensure_video_loaded()

@app.post("/admin/api/synonyms/generate", dependencies=[Depends(verify_admin)])
async def generate_synonyms(background_tasks: BackgroundTasks):
    """data.jsonã‹ã‚‰åŒç¾©èªžã‚’ç”Ÿæˆ"""
    await state.ensure_video_loaded()
    
    synonym_map = {}
    for v in state.videos:
        title = v.get("title", "")
        desc = v.get("description", "")
        
        # ã‚¿ã‚¤ãƒˆãƒ«ã‹ã‚‰ä¸»è¦ã‚­ãƒ¼ãƒ¯ãƒ¼ãƒ‰æŠ½å‡ºï¼ˆç°¡æ˜“ç‰ˆï¼‰
        words = re.findall(r'[\w]+', title + " " + desc)
        for word in words:
            if len(word) > 2:
                if word not in synonym_map:
                    synonym_map[word] = []
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonym_map, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "ok", "count": len(synonym_map)}

@app.patch("/admin/api/synonyms/{term}", dependencies=[Depends(verify_admin)])
async def update_synonym_term(term: str, values: List[str], background_tasks: BackgroundTasks):
    """åŒç¾©èªžã®å€‹åˆ¥æ›´æ–°"""
    synonyms = {}
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            synonyms = json.load(f)
    
    synonyms[term] = values
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "ok", "term": term}

@app.delete("/admin/api/synonyms/{term}", dependencies=[Depends(verify_admin)])
async def delete_synonym_term(term: str):
    """同義語の個別削除"""
    print(f"🗑️ Deleting synonym: {term}")
    synonyms = {}
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            synonyms = json.load(f)
    
    if term in synonyms:
        del synonyms[term]
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Synonym deleted successfully")
    return {"status": "ok", "term": term}

@app.get("/admin/api/faq", dependencies=[Depends(verify_admin)])
async def get_faq():
    """FAQå…¨ä½“å–å¾—"""
    if FAQ_PATH.exists():
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.put("/admin/api/faq", dependencies=[Depends(verify_admin)])
async def update_faq(data: dict, background_tasks: BackgroundTasks):
    """FAQå…¨ä½“æ›´æ–°"""
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "ok"}

async def reload_faq_data():
    """FAQãƒ‡ãƒ¼ã‚¿ã®ãƒªãƒ­ãƒ¼ãƒ‰"""
    state.faq_loaded = False
    await state.ensure_faq_loaded()

# ============================================
# ç®¡ç†API - FAQå€‹åˆ¥ç·¨é›†
# ============================================

@app.get("/admin/api/faq/items", dependencies=[Depends(verify_admin)])
async def list_faq_items(offset: int = 0, limit: int = 50, q: str = ""):
    """FAQä¸€è¦§å–å¾—ï¼ˆæ¤œç´¢ãƒ»ãƒšãƒ¼ã‚¸ãƒ³ã‚°å¯¾å¿œï¼‰"""
    await state.ensure_faq_loaded()
    
    items = state.faq_items_flat
    
    if q:
        q_lower = q.lower()
        items = [
            item for item in items
            if q_lower in item.get("question", "").lower()
            or q_lower in item.get("category", "").lower()
            or any(q_lower in kw.lower() for kw in item.get("keywords", []))
        ]
    
    total = len(items)
    page_items = items[offset:offset + limit]
    
    return {"items": page_items, "has_more": (offset + limit) < total, "total": total}

def generate_faq_id(category: str) -> str:
    """
    FAQ IDを自動採番する。
    「カテゴリ名-連番3桁」形式（例: パスワード-001）で、
    既存データと重複しない最小の連番を割り当てる。
    カテゴリ未指定時は「FAQ-連番」形式。
    """
    prefix = (category or "FAQ").strip() or "FAQ"

    existing_ids = {f.get("id", "") for f in state.faq_items_flat}

    max_num = 0
    pattern = re.compile(r"^" + re.escape(prefix) + r"-(\d+)$")
    for eid in existing_ids:
        m = pattern.match(eid or "")
        if m:
            max_num = max(max_num, int(m.group(1)))

    next_num = max_num + 1
    candidate = f"{prefix}-{next_num:03d}"

    # 念のため重複がなくなるまでインクリメント（並行作成などのレアケース対策）
    while candidate in existing_ids:
        next_num += 1
        candidate = f"{prefix}-{next_num:03d}"

    return candidate

# ------------------------------------------------------------
# FAQ作成・更新・削除の共通処理
# 管理画面API(/admin/api/faq/item, Basic認証)と
# 外部システムAPI(/api/v1/faq, APIキー認証)の両方から利用する。
# faqs配列形式・カテゴリ辞書形式（後方互換）の両方に対応。
# ------------------------------------------------------------

def _faq_id_matches(existing: dict, item_id: str) -> bool:
    """
    raw JSON上のFAQレコードは "id" と "faq_id" のどちらのキーで
    保存されている場合もあるため（インポート元データにより異なる）、両方を照合する。
    """
    return existing.get("id") == item_id or existing.get("faq_id") == item_id

def _faq_find_location(item_id: str):
    """item_idが格納されているリストとインデックスを探す。見つからなければNone。"""
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        for i, existing in enumerate(state.faq_data["faqs"]):
            if isinstance(existing, dict) and _faq_id_matches(existing, item_id):
                return (state.faq_data["faqs"], i)
        return None
    else:
        for category, items in state.faq_data.items():
            if isinstance(items, list):
                for i, existing in enumerate(items):
                    if isinstance(existing, dict) and _faq_id_matches(existing, item_id):
                        return (items, i)
    return None

def _faq_get_by_id(item_id: str):
    loc = _faq_find_location(item_id)
    if not loc:
        return None
    container, i = loc
    return container[i]

async def _create_faq_item_internal(item: dict, changed_by: str = "admin") -> str:
    """FAQ新規作成の共通処理。作成後のfaq_idを返す。"""
    await state.ensure_faq_loaded()

    faq_id = (item.get("id") or "").strip()
    if faq_id:
        if any(f.get("id") == faq_id for f in state.faq_items_flat):
            raise HTTPException(400, f"ID '{faq_id}' already exists")
    else:
        faq_id = generate_faq_id(item.get("category", ""))
        item = dict(item)
        item["id"] = faq_id

    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        state.faq_data["faqs"].append(item)
    else:
        category = item.get("category", "その他")
        if category not in state.faq_data:
            state.faq_data[category] = []
        state.faq_data[category].append(item)

    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)

    record_faq_history(faq_id, "create", changed_by, None, item)
    return faq_id

async def _update_faq_item_internal(item_id: str, item: dict, changed_by: str = "admin"):
    """FAQ更新の共通処理。見つからない場合は404を送出する。"""
    await state.ensure_faq_loaded()

    loc = _faq_find_location(item_id)
    if not loc:
        raise HTTPException(404, f"FAQ item '{item_id}' not found")

    container, i = loc
    before = dict(container[i]) if isinstance(container[i], dict) else None
    container[i] = item

    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)

    record_faq_history(item_id, "update", changed_by, before, item)

async def _delete_faq_item_internal(item_id: str, changed_by: str = "admin"):
    """FAQ削除の共通処理。見つからない場合は404を送出する。"""
    await state.ensure_faq_loaded()

    loc = _faq_find_location(item_id)
    if not loc:
        raise HTTPException(404, f"FAQ item '{item_id}' not found")

    container, i = loc
    before = dict(container[i]) if isinstance(container[i], dict) else None
    del container[i]

    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)

    record_faq_history(item_id, "delete", changed_by, before, None)


@app.post("/admin/api/faq/item", dependencies=[Depends(verify_admin)])
async def create_faq_item(item: dict, background_tasks: BackgroundTasks):
    """FAQ新規作成（管理画面）"""
    faq_id = await _create_faq_item_internal(item, changed_by="admin")
    background_tasks.add_task(reload_faq_data)
    return {"status": "created", "id": faq_id}

@app.patch("/admin/api/faq/item/{item_id}", dependencies=[Depends(verify_admin)])
async def update_faq_item(item_id: str, item: dict, background_tasks: BackgroundTasks):
    """FAQ更新（管理画面）"""
    await _update_faq_item_internal(item_id, item, changed_by="admin")
    background_tasks.add_task(reload_faq_data)
    return {"status": "updated", "id": item_id}

@app.delete("/admin/api/faq/item/{item_id}", dependencies=[Depends(verify_admin)])
async def delete_faq_item(item_id: str, background_tasks: BackgroundTasks):
    """FAQ削除（管理画面）"""
    await _delete_faq_item_internal(item_id, changed_by="admin")
    background_tasks.add_task(reload_faq_data)
    return {"status": "deleted", "id": item_id}

@app.delete("/admin/api/faq/all", dependencies=[Depends(verify_admin)])
async def delete_all_faqs(background_tasks: BackgroundTasks):
    """FAQ一括削除"""
    print("🗑️ FAQ bulk delete requested")
    await state.ensure_faq_loaded()
    
    # faqs配列形式
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        count = len(state.faq_data["faqs"])
        state.faq_data["faqs"] = []
        print(f"✅ Deleted {count} FAQs (faqs array format)")
    else:
        # カテゴリ辞書形式
        count = sum(len(items) for items in state.faq_data.values() if isinstance(items, list))
        for key in list(state.faq_data.keys()):
            if isinstance(state.faq_data[key], list):
                state.faq_data[key] = []
        print(f"✅ Deleted {count} FAQs (category dict format)")
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "deleted", "count": count}

@app.post("/admin/api/faq/import", dependencies=[Depends(verify_admin)])
async def import_faqs(data: dict, background_tasks: BackgroundTasks):
    """FAQインポート（新規追加・更新のみ）"""
    print("📤 FAQ import requested")
    await state.ensure_faq_loaded()
    
    imported_faqs = data.get("faqs", [])
    if not isinstance(imported_faqs, list):
        print(f"❌ Invalid format: faqs is {type(imported_faqs)}")
        raise HTTPException(400, "Invalid format: 'faqs' must be an array")
    
    print(f"📋 Importing {len(imported_faqs)} FAQs")
    added_count = 0
    updated_count = 0
    
    # faqs配列形式
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        existing_ids = {item.get("id") for item in state.faq_data["faqs"] if isinstance(item, dict)}
        print(f"   Existing FAQ IDs: {len(existing_ids)}")
        
        for imported_item in imported_faqs:
            if not isinstance(imported_item, dict):
                continue
            
            item_id = imported_item.get("id") or imported_item.get("faq_id")
            if not item_id:
                print(f"   ⚠️ Skipping item without ID")
                continue
            
            # フィールド正規化
            normalized_item = imported_item.copy()
            if "faq_id" in normalized_item and "id" not in normalized_item:
                normalized_item["id"] = normalized_item.pop("faq_id")
            if "answer_steps" in normalized_item and "steps" not in normalized_item:
                normalized_item["steps"] = normalized_item.pop("answer_steps")
            
            if item_id in existing_ids:
                # 更新
                for i, existing in enumerate(state.faq_data["faqs"]):
                    if existing.get("id") == item_id:
                        state.faq_data["faqs"][i] = normalized_item
                        updated_count += 1
                        print(f"   ✏️ Updated: {item_id}")
                        break
            else:
                # 新規追加
                state.faq_data["faqs"].append(normalized_item)
                added_count += 1
                print(f"   ➕ Added: {item_id}")
    else:
        # カテゴリ辞書形式への対応
        for imported_item in imported_faqs:
            if not isinstance(imported_item, dict):
                continue
            
            item_id = imported_item.get("id")
            if not item_id:
                continue
            
            category = imported_item.get("category", "その他")
            
            # カテゴリが存在しない場合は作成
            if category not in state.faq_data:
                state.faq_data[category] = []
            
            # 既存チェック
            found = False
            if isinstance(state.faq_data[category], list):
                for i, existing in enumerate(state.faq_data[category]):
                    if existing.get("id") == item_id:
                        state.faq_data[category][i] = imported_item
                        updated_count += 1
                        found = True
                        break
            
            if not found:
                state.faq_data[category].append(imported_item)
                added_count += 1
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {
        "status": "imported",
        "added": added_count,
        "updated": updated_count,
        "total": added_count + updated_count
    }

@app.get("/admin/api/faq/export", dependencies=[Depends(verify_admin)])
async def export_faqs():
    """FAQエクスポート"""
    print("📥 FAQ export requested")
    await state.ensure_faq_loaded()
    
    # フィールド名を元に戻す関数
    def normalize_for_export(faq):
        """エクスポート用にフィールド名を元に戻す"""
        exported_faq = faq.copy()
        
        # id → faq_id
        if "id" in exported_faq:
            exported_faq["faq_id"] = exported_faq.pop("id")
        
        # steps → answer_steps
        if "steps" in exported_faq:
            exported_faq["answer_steps"] = exported_faq.pop("steps")
        
        return exported_faq
    
    # faqs配列形式で返す
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        # 各FAQのフィールド名を元に戻す
        exported_faqs = [normalize_for_export(faq) for faq in state.faq_data["faqs"]]
        
        export_data = {
            "meta": state.faq_data.get("meta", {}),
            "faqs": exported_faqs
        }
        print(f"✅ Exporting {len(exported_faqs)} FAQs (faqs array format)")
    else:
        # カテゴリ辞書形式をfaqs配列形式に変換
        all_faqs = []
        for category, items in state.faq_data.items():
            if isinstance(items, list):
                all_faqs.extend(items)
        
        # 各FAQのフィールド名を元に戻す
        exported_faqs = [normalize_for_export(faq) for faq in all_faqs]
        
        import time
        export_data = {
            "meta": {
                "exported_at": time.time(),
                "count": len(exported_faqs)
            },
            "faqs": exported_faqs
        }
        print(f"✅ Exporting {len(exported_faqs)} FAQs (category dict format)")
    
    return export_data

@app.get("/admin/api/faq/export/template", dependencies=[Depends(verify_admin)])
async def export_faq_template():
    """
    新規追加・更新用の空テンプレートJSONをエクスポート
    項目（キー）のみ入ったサンプル1件を含む、記入用フォーマット
    """
    print("📄 FAQ template export requested")

    template_data = {
        "meta": {
            "description": "このファイルはFAQインポート用のテンプレートです。faqsの配列にFAQを追加してインポートしてください。",
            "note_id": "id を指定すると更新、省略すると新規作成（自動採番）されます。",
            "note_language": "language は ja または en を指定してください。"
        },
        "faqs": [
            {
                "id": "",
                "category": "",
                "language": "ja",
                "question": "",
                "answer": "",
                "steps": [
                    ""
                ],
                "note": "",
                "keywords": [
                    ""
                ]
            }
        ]
    }

    return template_data



# ============================================
# ç®¡ç†API - å‹•ç”»ãƒ‡ãƒ¼ã‚¿
# ============================================

@app.get("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def get_videos():
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ä¸€è¦§å–å¾—"""
    if not DATA_PATH.exists():
        return []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    return videos

@app.post("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def create_video(video_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ä½œæˆ"""
    videos = []
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            videos = json.load(f)
    
    videos.append(video_data)
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "created", "video_id": video_data.get("video_id")}

@app.patch("/admin/api/videos/{video_id}", dependencies=[Depends(verify_admin)])
async def update_video(video_id: str, video_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿æ›´æ–°"""
    if not DATA_PATH.exists():
        raise HTTPException(404, "Data file not found")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    found = False
    for i, video in enumerate(videos):
        if video.get("video_id") == video_id:
            videos[i] = video_data
            found = True
            break
    
    if not found:
        raise HTTPException(404, f"Video '{video_id}' not found")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "updated", "video_id": video_id}

@app.put("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def update_videos(videos: List[dict], background_tasks: BackgroundTasks):
    """動画データ一括更新（言語タグ編集用）"""
    print(f"💾 Updating videos: {len(videos)} items")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Videos saved successfully")
    
    # FAISSインデックスを再構築
    background_tasks.add_task(reload_video_data)
    
    return {"status": "ok", "count": len(videos)}

@app.delete("/admin/api/videos/{video_id}", dependencies=[Depends(verify_admin)])
async def delete_video(video_id: str, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿å‰Šé™¤"""
    if not DATA_PATH.exists():
        raise HTTPException(404, "Data file not found")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    found = False
    for i, video in enumerate(videos):
        if video.get("video_id") == video_id:
            del videos[i]
            found = True
            break
    
    if not found:
        raise HTTPException(404, f"Video '{video_id}' not found")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "deleted", "video_id": video_id}

@app.post("/admin/api/videos/bulk-delete", dependencies=[Depends(verify_admin)])
async def bulk_delete_videos(request_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ä¸€æ‹¬å‰Šé™¤"""
    video_ids = request_data.get("video_ids", [])
    
    if not video_ids:
        raise HTTPException(400, "video_ids is required")
    
    if not DATA_PATH.exists():
        raise HTTPException(404, "Data file not found")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    # å‰Šé™¤å¯¾è±¡ä»¥å¤–ã‚’æ®‹ã™
    filtered_videos = [v for v in videos if v.get('video_id') not in video_ids]
    
    # noç•ªå·ã‚’æŒ¯ã‚Šç›´ã—
    for i, video in enumerate(filtered_videos, 1):
        video['no'] = i
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    
    return {
        "status": "success",
        "deleted_count": len(video_ids),
        "remaining_count": len(filtered_videos)
    }

@app.post("/admin/api/videos/delete-all", dependencies=[Depends(verify_admin)])
@app.delete("/admin/api/videos/all", dependencies=[Depends(verify_admin)])
async def delete_all_videos(background_tasks: BackgroundTasks):
    """data.jsonå…¨å‰Šé™¤ï¼ˆå®Œå…¨ãƒªã‚»ãƒƒãƒˆï¼‰"""
    if not DATA_PATH.exists():
        # data.jsonãŒãªã„å ´åˆã‚‚ç©ºé…åˆ—ã‚’ä½œæˆ
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {
            "status": "success",
            "message": "Data file created as empty array"
        }
    
    # ç©ºã®é…åˆ—ã§ä¸Šæ›¸ã
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    
    return {
        "status": "success",
        "message": "All video data deleted"
    }

@app.post("/admin/api/videos/import", dependencies=[Depends(verify_admin)])
async def import_videos(import_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ã‚¤ãƒ³ãƒãƒ¼ãƒˆ"""
    mode = import_data.get("mode", "merge")
    new_data = import_data.get("data", [])
    
    if not isinstance(new_data, list):
        raise HTTPException(400, "Invalid data format")
    
    added = 0
    updated = 0
    
    if mode == "replace":
        # å…¨ä½“ç½®æ›
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        added = len(new_data)
    else:
        # å·®åˆ†ãƒžãƒ¼ã‚¸
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        existing_ids = {v.get("video_id"): i for i, v in enumerate(existing_videos)}
        
        for new_video in new_data:
            video_id = new_video.get("video_id")
            if video_id in existing_ids:
                existing_videos[existing_ids[video_id]] = new_video
                updated += 1
            else:
                existing_videos.append(new_video)
                added += 1
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "imported", "added": added, "updated": updated}

# reload_video_data defined above

# ============================================
# ç®¡ç†API - YouTubeæ–‡å­—èµ·ã“ã—
# ============================================

@app.post("/admin/api/youtube/fetch", dependencies=[Depends(verify_admin)])
async def fetch_youtube_videos(request_data: dict):
    """YouTubeãƒãƒ£ãƒ³ãƒãƒ«ã‹ã‚‰å‹•ç”»ãƒªã‚¹ãƒˆã‚’å–å¾—"""
    try:
        from googleapiclient.discovery import build
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(400, "YOUTUBE_API_KEY environment variable not set")
        
        channel_url = request_data.get("channel_url", "")
        max_results = request_data.get("max_results", 50)
        
        # ãƒãƒ£ãƒ³ãƒãƒ«IDã‚’æŠ½å‡º
        channel_id = None
        if "/c/" in channel_url or "/channel/" in channel_url or "/@" in channel_url:
            # ãƒãƒ£ãƒ³ãƒãƒ«åã‹ã‚‰IDã‚’å–å¾—ã™ã‚‹å¿…è¦ãŒã‚ã‚‹
            # ç°¡ç•¥åŒ–ã®ãŸã‚ã€ãƒ¦ãƒ¼ã‚¶ãƒ¼ã«ãƒãƒ£ãƒ³ãƒãƒ«IDã‚’ç›´æŽ¥å…¥åŠ›ã—ã¦ã‚‚ã‚‰ã†æ–¹å¼ã‚‚æ¤œè¨Ž
            parts = channel_url.rstrip('/').split('/')
            channel_name = parts[-1]
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            # ãƒãƒ£ãƒ³ãƒãƒ«åã‹ã‚‰æ¤œç´¢
            search_response = youtube.search().list(
                q=channel_name,
                type='channel',
                part='id',
                maxResults=1
            ).execute()
            
            if search_response.get('items'):
                channel_id = search_response['items'][0]['id']['channelId']
        else:
            raise HTTPException(400, "Invalid channel URL format")
        
        if not channel_id:
            raise HTTPException(404, "Channel not found")
        
        # ãƒãƒ£ãƒ³ãƒãƒ«ã®ã‚¢ãƒƒãƒ—ãƒ­ãƒ¼ãƒ‰ãƒ—ãƒ¬ã‚¤ãƒªã‚¹ãƒˆIDã‚’å–å¾—
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channel_response.get('items'):
            raise HTTPException(404, "Channel not found")
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # ãƒ—ãƒ¬ã‚¤ãƒªã‚¹ãƒˆã‹ã‚‰å‹•ç”»ã‚’å–å¾—
        videos = []
        next_page_token = None
        
        while len(videos) < max_results:
            playlist_response = youtube.playlistItems().list(
                playlistId=uploads_playlist_id,
                part='snippet',
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response.get('items', []):
                snippet = item['snippet']
                video_id = snippet['resourceId']['videoId']
                
                videos.append({
                    'video_id': video_id,
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'thumbnail': snippet['thumbnails'].get('high', {}).get('url', ''),
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'published_at': snippet['publishedAt']
                })
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        return {
            'status': 'success',
            'channel_id': channel_id,
            'videos': videos,
            'total': len(videos)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch YouTube videos: {str(e)}")

@app.post("/admin/api/youtube/transcribe", dependencies=[Depends(verify_admin)])
async def transcribe_youtube_video(request_data: dict, background_tasks: BackgroundTasks):
    """YouTubeå‹•ç”»ã‚’æ–‡å­—èµ·ã“ã—"""
    try:
        video_id = request_data.get("video_id")
        if not video_id:
            raise HTTPException(400, "video_id is required")
        
        # æ—¢å­˜ã®data.jsonã‚’èª­ã¿è¾¼ã¿
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # æ—¢ã«å­˜åœ¨ã™ã‚‹ã‹ãƒã‚§ãƒƒã‚¯
        for video in existing_videos:
            if video.get('video_id') == video_id:
                return {
                    'status': 'already_exists',
                    'message': 'Video already transcribed',
                    'video_id': video_id
                }
        
        # ãƒãƒƒã‚¯ã‚°ãƒ©ã‚¦ãƒ³ãƒ‰ã§æ–‡å­—èµ·ã“ã—å‡¦ç†
        background_tasks.add_task(process_transcription, video_id, request_data)
        
        return {
            'status': 'processing',
            'message': 'Transcription started in background',
            'video_id': video_id
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to start transcription: {str(e)}")

@app.post("/admin/api/youtube/sync", dependencies=[Depends(verify_admin)])
async def sync_with_youtube(request_data: dict):
    """YouTubeãƒãƒ£ãƒ³ãƒãƒ«ã¨data.jsonã‚’åŒæœŸï¼ˆå·®åˆ†æ¤œå‡ºï¼‰"""
    try:
        from googleapiclient.discovery import build
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(400, "YOUTUBE_API_KEY environment variable not set")
        
        channel_url = request_data.get("channel_url", "")
        
        # ãƒãƒ£ãƒ³ãƒãƒ«IDã‚’æŠ½å‡º
        channel_id = None
        if "/c/" in channel_url or "/channel/" in channel_url or "/@" in channel_url:
            parts = channel_url.rstrip('/').split('/')
            channel_name = parts[-1]
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            search_response = youtube.search().list(
                q=channel_name,
                type='channel',
                part='id',
                maxResults=1
            ).execute()
            
            if search_response.get('items'):
                channel_id = search_response['items'][0]['id']['channelId']
        
        if not channel_id:
            raise HTTPException(404, "Channel not found")
        
        # ãƒãƒ£ãƒ³ãƒãƒ«ã®å…¨å‹•ç”»ã‚’å–å¾—
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channel_response.get('items'):
            raise HTTPException(404, "Channel not found")
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # ãƒ—ãƒ¬ã‚¤ãƒªã‚¹ãƒˆã‹ã‚‰å…¨å‹•ç”»ã‚’å–å¾—
        youtube_videos = []
        next_page_token = None
        
        while True:
            playlist_response = youtube.playlistItems().list(
                playlistId=uploads_playlist_id,
                part='snippet',
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response.get('items', []):
                snippet = item['snippet']
                video_id = snippet['resourceId']['videoId']
                
                youtube_videos.append({
                    'video_id': video_id,
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'thumbnail': snippet['thumbnails'].get('high', {}).get('url', ''),
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'published_at': snippet['publishedAt']
                })
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        # æ—¢å­˜ã®data.jsonã‚’èª­ã¿è¾¼ã¿
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # å·®åˆ†ã‚’è¨ˆç®—
        youtube_ids = set(v['video_id'] for v in youtube_videos)
        existing_ids = set(v.get('video_id') for v in existing_videos)
        
        # YouTubeã«ã‚ã‚‹ãŒã€data.jsonã«ãªã„ï¼ˆè¿½åŠ ã™ã¹ãå‹•ç”»ï¼‰
        missing_in_data = [v for v in youtube_videos if v['video_id'] not in existing_ids]
        
        # data.jsonã«ã‚ã‚‹ãŒã€YouTubeã«ãªã„ï¼ˆå‰Šé™¤ã™ã¹ãå‹•ç”»ï¼‰
        missing_in_youtube = [v for v in existing_videos if v.get('video_id') not in youtube_ids]
        
        return {
            'status': 'success',
            'total_youtube': len(youtube_videos),
            'total_data': len(existing_videos),
            'missing_in_data': missing_in_data,
            'missing_in_youtube': missing_in_youtube,
            'youtube_ids': list(youtube_ids),
            'existing_ids': list(existing_ids)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to sync with YouTube: {str(e)}")

@app.post("/admin/api/youtube/cleanup", dependencies=[Depends(verify_admin)])
async def cleanup_orphaned_videos(request_data: dict, background_tasks: BackgroundTasks):
    """YouTubeã«å­˜åœ¨ã—ãªã„å‹•ç”»ã‚’data.jsonã‹ã‚‰å‰Šé™¤"""
    try:
        video_ids_to_delete = request_data.get("video_ids", [])
        
        if not video_ids_to_delete:
            raise HTTPException(400, "video_ids is required")
        
        if not DATA_PATH.exists():
            raise HTTPException(404, "Data file not found")
        
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            videos = json.load(f)
        
        # å‰Šé™¤å¯¾è±¡ä»¥å¤–ã®å‹•ç”»ã‚’æ®‹ã™
        filtered_videos = [v for v in videos if v.get('video_id') not in video_ids_to_delete]
        
        # noç•ªå·ã‚’æŒ¯ã‚Šç›´ã—
        for i, video in enumerate(filtered_videos, 1):
            video['no'] = i
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(filtered_videos, f, ensure_ascii=False, indent=2)
        
        background_tasks.add_task(reload_video_data)
        
        return {
            'status': 'success',
            'deleted_count': len(video_ids_to_delete),
            'remaining_count': len(filtered_videos)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to cleanup: {str(e)}")

async def process_transcription(video_id: str, video_data: dict):
    """æ–‡å­—èµ·ã“ã—å‡¦ç†ï¼ˆãƒãƒƒã‚¯ã‚°ãƒ©ã‚¦ãƒ³ãƒ‰ï¼‰- yt-dlp Pythonãƒ©ã‚¤ãƒ–ãƒ©ãƒªä½¿ç”¨"""
    import tempfile
    import os
    import yt_dlp
    
    audio_path = None
    
    try:
        print(f"[INFO] Starting transcription for {video_id}: {video_data.get('title', '')}")
        
        # 1. éŸ³å£°ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰ï¼ˆyt-dlp Pythonãƒ©ã‚¤ãƒ–ãƒ©ãƒªã‚’ä½¿ç”¨ï¼‰
        temp_dir = tempfile.gettempdir()
        output_basename = f"audio_{video_id}"
        output_path = os.path.join(temp_dir, output_basename)
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        
        print(f"[INFO] Downloading audio from {video_url}")
        
        # yt-dlpè¨­å®šï¼ˆbotæ¤œå‡ºå›žé¿ã‚’å«ã‚€ï¼‰
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path + '.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # botæ¤œå‡ºå›žé¿: iOSã‚¯ãƒ©ã‚¤ã‚¢ãƒ³ãƒˆã‚’ä½¿ç”¨
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'web']
                }
            },
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'postprocessor_args': ['-ar', '16000'],
            'prefer_ffmpeg': True,
        }
        
        # yt-dlpã§ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] yt-dlp download failed: {error_msg}")
            
            # botæ¤œå‡ºã‚¨ãƒ©ãƒ¼ã®åˆ¤å®š
            if 'Sign in to confirm' in error_msg or 'bot' in error_msg.lower():
                raise Exception(
                    "YouTube botæ¤œå‡º: ã“ã®å‹•ç”»ã¯ç¾åœ¨ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰ã§ãã¾ã›ã‚“ã€‚"
                )
            
            raise Exception(f"yt-dlp download failed: {error_msg[:300]}")
        
        # MP3ãƒ•ã‚¡ã‚¤ãƒ«ã®å­˜åœ¨ç¢ºèª
        audio_path = output_path + '.mp3'
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not created: {audio_path}")
        
        print(f"[INFO] Audio downloaded to {audio_path}")
        
        # 2. Whisperã§æ–‡å­—èµ·ã“ã—
        try:
            import whisper
            print(f"[INFO] Loading Whisper model...")
            model = whisper.load_model("base")
            print(f"[INFO] Transcribing...")
            result = model.transcribe(audio_path, language='ja', fp16=False)
            transcript = result['text']
            print(f"[INFO] Transcription completed: {len(transcript)} characters")
        except ImportError:
            raise Exception("Whisper not installed. Please install: pip install openai-whisper")
        except Exception as e:
            raise Exception(f"Whisper transcription failed: {str(e)}")
        
        # 3. data.jsonã«è¿½åŠ 
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # æ–°ã—ã„noç•ªå·ã‚’ç”Ÿæˆ
        max_no = max([v.get('no', 0) for v in existing_videos], default=0)
        
        new_video = {
            'no': max_no + 1,
            'video_id': video_id,
            'title': video_data.get('title', ''),
            'description': video_data.get('description', ''),
            'transcript': transcript,
            'url': video_data.get('url', ''),
            'thumbnail': video_data.get('thumbnail', ''),
            'status': 'completed'
        }
        
        existing_videos.append(new_video)
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_videos, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] Transcription saved for {video_id}")
        
        # å‹•ç”»ãƒ‡ãƒ¼ã‚¿å†èª­ã¿è¾¼ã¿
        await reload_video_data()
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Transcription error for {video_id}: {error_msg}")
        
        # ã‚¨ãƒ©ãƒ¼æ™‚ã‚‚data.jsonã«è¨˜éŒ²ï¼ˆstatus: failedï¼‰
        try:
            existing_videos = []
            if DATA_PATH.exists():
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    existing_videos = json.load(f)
            
            max_no = max([v.get('no', 0) for v in existing_videos], default=0)
            
            # ã‚¨ãƒ©ãƒ¼ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ã‚’åˆ†ã‹ã‚Šã‚„ã™ãå¤‰æ›
            friendly_error = error_msg
            if 'bot' in error_msg.lower() or 'Sign in to confirm' in error_msg:
                friendly_error = "YouTube botæ¤œå‡º: ã“ã®å‹•ç”»ã¯ç¾åœ¨ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰ã§ãã¾ã›ã‚“ã€‚ã—ã°ã‚‰ãå¾…ã£ã¦ã‹ã‚‰å†è©¦è¡Œã—ã¦ãã ã•ã„ã€‚"
            elif 'JavaScript runtime' in error_msg:
                friendly_error = "JavaScriptå‡¦ç†ã‚¨ãƒ©ãƒ¼: ã“ã®å‹•ç”»ã¯ç‰¹æ®Šãªå‡¦ç†ãŒå¿…è¦ã§ã™ã€‚YouTube Data APIã‹ã‚‰å–å¾—ã—ãŸå‹•ç”»æƒ…å ±ã®ã¿ä¿å­˜ã•ã‚Œã¾ã™ã€‚"
            
            error_video = {
                'no': max_no + 1,
                'video_id': video_id,
                'title': video_data.get('title', ''),
                'description': video_data.get('description', ''),
                'transcript': f'æ–‡å­—èµ·ã“ã—å¤±æ•—: {friendly_error}',
                'url': video_data.get('url', ''),
                'thumbnail': video_data.get('thumbnail', ''),
                'status': 'failed'
            }
            
            existing_videos.append(error_video)
            
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(existing_videos, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] Error status saved for {video_id}")
        except Exception as save_error:
            print(f"[ERROR] Failed to save error status: {str(save_error)}")
    
    finally:
        # ä¸€æ™‚ãƒ•ã‚¡ã‚¤ãƒ«å‰Šé™¤
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"[INFO] Temporary file removed: {audio_path}")
            except Exception as cleanup_error:
                print(f"[WARNING] Failed to remove temp file: {str(cleanup_error)}")

# ============================================
# ç®¡ç†API - ãƒ­ã‚°
# ============================================

@app.get("/admin/api/logs/debug", dependencies=[Depends(verify_admin)])
async def debug_logs():
    """ログデバッグ情報（問題調査用）"""
    info = {
        "data_dir":          str(DATA_DIR),
        "search_log_path":   str(SEARCH_LOG_PATH),
        "file_exists":       SEARCH_LOG_PATH.exists(),
        "file_size_bytes":   0,
        "file_first_5_lines": [],
        "parsed_row_count":  0,
        "sample_rows":       [],
        "error":             None,
    }
    try:
        if SEARCH_LOG_PATH.exists():
            info["file_size_bytes"] = SEARCH_LOG_PATH.stat().st_size
            with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                info["file_first_5_lines"] = [l.rstrip() for l in lines[:5]]

        rows = parse_logs()
        info["parsed_row_count"] = len(rows)
        info["sample_rows"] = [
            {
                "dt":          r["dt"].isoformat(),
                "result_type": r["result_type"],
                "query":       r["query"],
                "result_id":   r["result_id"],
            }
            for r in rows[:5]
        ]
    except Exception as e:
        info["error"] = str(e)

    return info

@app.get("/admin/api/logs/months", dependencies=[Depends(verify_admin)])
async def get_log_months():
    """利用可能な月一覧"""
    rows   = parse_logs()
    months = sorted(set(r["dt"].strftime("%Y-%m") for r in rows), reverse=True)
    return {"months": months, "total": len(rows)}

@app.get("/admin/api/logs/summary", dependencies=[Depends(verify_admin)])
async def get_log_summary(month: str = Query(...)):
    """月別サマリー（type別集計付き）"""
    rows = parse_logs()

    day_counter   = Counter()
    faq_counter   = Counter()
    video_counter = Counter()
    faq_count     = 0
    video_count   = 0

    for r in rows:
        if r["dt"].strftime("%Y-%m") != month:
            continue
        day_counter[r["dt"].strftime("%Y-%m-%d")] += 1
        rtype = r.get("result_type", "")
        query = r.get("query", "")
        if rtype == "faq":
            faq_counter[query] += 1
            faq_count += 1
        elif rtype == "video":
            video_counter[query] += 1
            video_count += 1
        else:
            faq_counter[query] += 1

    days      = [{"day": d, "count": c} for d, c in sorted(day_counter.items())]
    top_faq   = [{"query": q, "count": c} for q, c in faq_counter.most_common(50)]
    top_video = [{"query": q, "count": c} for q, c in video_counter.most_common(50)]
    total     = sum(day_counter.values())

    return {
        "month":       month,
        "total":       total,
        "faq_count":   faq_count,
        "video_count": video_count,
        "days":        days,
        "top_faq":     top_faq,
        "top_video":   top_video,
    }


@app.get("/admin/api/logs/export", dependencies=[Depends(verify_admin)])
async def export_logs():
    """ログCSVエクスポート"""
    if not SEARCH_LOG_PATH.exists():
        raise HTTPException(404, "No logs found")
    
    csv_data = SEARCH_LOG_PATH.read_text(encoding="utf-8")
    
    # Excelで開いたときに日本語が文字化けしないよう、UTF-8 BOMを付与する
    bom = "\ufeff"
    csv_bytes = (bom + csv_data).encode("utf-8")
    
    headers = {"Content-Disposition": 'attachment; filename="search_logs.csv"'}
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv; charset=utf-8",
        headers=headers
    )

# ============================================
# é™çš„ãƒ•ã‚¡ã‚¤ãƒ«é…ä¿¡
# ============================================

# ãƒ•ãƒ­ãƒ³ãƒˆã‚¨ãƒ³ãƒ‰é™çš„ãƒ•ã‚¡ã‚¤ãƒ«
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ç®¡ç†ç”»é¢é™çš„ãƒ•ã‚¡ã‚¤ãƒ«
app.mount("/admin/static", StaticFiles(directory=admin_path), name="admin_static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_index():
    """æ¤œç´¢ç”»é¢"""
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return index_file.read_text(encoding="utf-8")


@app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
def serve_debug():
    """診断ページ"""
    debug_file = frontend_path / "debug.html"
    if not debug_file.exists():
        return HTMLResponse("<h1>debug.html not found</h1>", status_code=404)
    return debug_file.read_text(encoding="utf-8")

@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_home():
    """ç®¡ç†ç”»é¢ãƒˆãƒƒãƒ—"""
    f = admin_path / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>admin index.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/dashboard", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_dashboard():
    """ç®¡ç†ç”»é¢ - ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰"""
    f = admin_path / "dashboard.html"
    if not f.exists():
        return HTMLResponse("<h1>admin dashboard.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/videos", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_videos():
    """ç®¡ç†ç”»é¢ - å‹•ç”»ãƒ‡ãƒ¼ã‚¿"""
    f = admin_path / "videos.html"
    if not f.exists():
        return HTMLResponse("<h1>admin videos.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/synonyms", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_synonyms():
    """ç®¡ç†ç”»é¢ - Synonyms"""
    f = admin_path / "synonyms.html"
    if not f.exists():
        return HTMLResponse("<h1>admin synonyms.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/faq", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq():
    """ç®¡ç†ç”»é¢ - FAQ"""
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/logs", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs():
    """ç®¡ç†ç”»é¢ - ãƒ­ã‚°"""
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)

    return f.read_text(encoding="utf-8")
@app.get("/admin/files", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_files():
    """ファイル管理ページ"""
    f = admin_path / "files.html"
    if not f.exists():
        return HTMLResponse("<h1>files.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/password", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_password():
    """管理画面 - パスワード変更"""
    f = admin_path / "password.html"
    if not f.exists():
        return HTMLResponse("<h1>admin password.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/apikeys", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_apikeys():
    """管理画面 - 外部連携APIキー管理"""
    f = admin_path / "apikeys.html"
    if not f.exists():
        return HTMLResponse("<h1>admin apikeys.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

    return f.read_text(encoding="utf-8")

# .htmlæ‹¡å¼µå­ä»˜ãã®ãƒ«ãƒ¼ãƒˆã‚‚è¿½åŠ 
@app.get("/admin/dashboard.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_dashboard_html():
    """ç®¡ç†ç”»é¢ - ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰ (.html)"""
    f = admin_path / "dashboard.html"
    if not f.exists():
        return HTMLResponse("<h1>admin dashboard.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/videos.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_videos_html():
    """ç®¡ç†ç”»é¢ - å‹•ç”»ãƒ‡ãƒ¼ã‚¿ (.html)"""
    f = admin_path / "videos.html"
    if not f.exists():
        return HTMLResponse("<h1>admin videos.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/synonyms.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_synonyms_html():
    """ç®¡ç†ç”»é¢ - Synonyms (.html)"""
    f = admin_path / "synonyms.html"
    if not f.exists():
        return HTMLResponse("<h1>admin synonyms.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/faq.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq_html():
    """ç®¡ç†ç”»é¢ - FAQ (.html)"""
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/logs.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs_html():
    """ç®¡ç†ç”»é¢ - ãƒ­ã‚° (.html)"""
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

# ============================================
# æ¤œç´¢ãƒ­ã‚°APIãƒ»Synonyms APIï¼ˆè¿½åŠ ï¼‰
# ============================================

@app.post("/api/log_search")
async def log_search_api(log_data: dict):
    """
    クリックログを記録（FAQ/動画をクリックした時に呼ばれる）
    CSVフォーマット: timestamp, type, query, result_id
    """
    try:
        timestamp   = log_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        result_type = str(log_data.get("result_type") or "")
        query       = str(log_data.get("query") or "")
        result_id   = str(log_data.get("result_id") or "")

        with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, result_type, query, result_id])

        print(f"📝 Log: type={result_type} query={query} id={result_id}")
    except Exception as e:
        print(f"⚠️ Log write failed: {e}")

    return {"status": "logged"}

@app.get("/api/ranking/faq")
async def get_faq_ranking(limit: int = 10):
    """FAQクリックランキング（CSVログから集計）"""
    rows = parse_logs()
    counter = Counter(
        r["result_id"] for r in rows
        if r.get("result_type") == "faq" and r.get("result_id")
    )
    await state.ensure_faq_loaded()
    ranking = []
    for faq_id, count in counter.most_common(limit):
        faq_item = next((item for item in state.faq_items_flat if item.get("id") == faq_id), None)
        if faq_item:
            ranking.append({
                "id": faq_id,
                "question": faq_item.get("question", ""),
                "category": faq_item.get("category", ""),
                "click_count": count
            })
    return {"ranking": ranking}

@app.get("/api/ranking/video")
async def get_video_ranking(limit: int = 10):
    """動画クリックランキング（CSVログから集計）"""
    rows = parse_logs()
    counter = Counter(
        r["result_id"] for r in rows
        if r.get("result_type") == "video" and r.get("result_id")
    )
    await state.ensure_video_loaded()
    ranking = []
    for video_id, count in counter.most_common(limit):
        video_item = next((item for item in state.videos if item.get("video_id") == video_id), None)
        if video_item:
            ranking.append({
                "video_id": video_id,
                "title": video_item.get("title", ""),
                "thumbnail": video_item.get("thumbnail", ""),
                "click_count": count
            })
    return {"ranking": ranking}


# ============================================
# 管理者認証API（パスワード + 認証アプリ(TOTP) による多要素認証）
# ============================================
# ログインの流れ:
#   1) POST /admin/api/auth/login  … ユーザー名+パスワードを確認
#        MFA登録済み   → status="mfa_required"        (mfa_token を返す)
#        MFA未登録     → status="mfa_setup_required"  (setup_token を返す)
#   2a) POST /admin/api/auth/mfa           … 6桁コード or リカバリーコードで本人確認 → session token
#   2b) POST /admin/api/auth/setup/start   … QRコード/シークレットを発行
#       POST /admin/api/auth/setup/confirm … コード確認 → MFA有効化 + リカバリーコード発行 + session token
# 以降の管理API呼び出しは「Authorization: Bearer <session token>」で行う。

GENERIC_LOGIN_ERROR = "ユーザー名またはパスワードが正しくありません"

def _lock_guard(username: str, ip: str):
    remain = au.lock_remaining(username, ip)
    if remain > 0:
        raise HTTPException(
            status_code=429,
            detail=f"ログイン失敗が続いたため一時的にロックしています。約{(remain + 59) // 60}分後に再度お試しください",
        )

def _session_response(user: dict) -> dict:
    token = au.issue_token(user["username"], "session", user.get("token_epoch", 1), au.SESSION_TTL_SECONDS)
    return {
        "status": "ok",
        "token": token,
        "username": user["username"],
        "must_change_password": bool(user.get("must_change_password")),
        "recovery_codes_remaining": len(user.get("recovery_codes", [])),
    }

@app.post("/admin/api/auth/login")
async def auth_login(body: dict, request: Request):
    """ステップ1: ユーザー名とパスワードの確認"""
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    ip = _client_ip(request)
    _lock_guard(username, ip)

    data = au.load_users()
    user = au.find_user(data, username)
    # ユーザーが存在しなくても同じ時間をかけて照合する（存在有無を推測されないため）
    stored = user.get("password_hash", "") if user else au._DUMMY_HASH
    ok = au.verify_password_hash(stored, password) and user is not None
    if not ok:
        au.record_failure(username, ip)
        print(f"❌ Login failed (password): user={username!r} ip={ip}")
        raise HTTPException(status_code=401, detail=GENERIC_LOGIN_ERROR)

    epoch = user.get("token_epoch", 1)
    if user.get("mfa_enabled"):
        return {"status": "mfa_required",
                "mfa_token": au.issue_token(username, "mfa", epoch, au.PENDING_TTL_SECONDS)}
    return {"status": "mfa_setup_required",
            "setup_token": au.issue_token(username, "setup", epoch, au.PENDING_TTL_SECONDS)}

@app.post("/admin/api/auth/mfa")
async def auth_mfa(body: dict, request: Request):
    """ステップ2: 認証アプリの6桁コード（またはリカバリーコード）で本人確認"""
    ip = _client_ip(request)
    payload = au.parse_token(str(body.get("mfa_token", "")), "mfa")
    if not payload:
        raise HTTPException(status_code=401, detail="確認の有効期限が切れました。最初からログインし直してください")
    username = payload["u"]
    _lock_guard(username, ip)

    code = str(body.get("code", "")).strip()
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        if not user or user.get("token_epoch", 1) != payload.get("e") or not user.get("mfa_enabled"):
            raise HTTPException(status_code=401, detail="確認の有効期限が切れました。最初からログインし直してください")

        verified = False
        if code.replace(" ", "").isdigit():
            step = au.verify_totp(user["totp_secret"], code, user.get("totp_last_step", 0))
            if step is not None:
                user["totp_last_step"] = step   # 同じコードの再利用を防ぐ
                verified = True
        elif code:
            verified = au.consume_recovery_code(user, code)   # リカバリーコードは1回限り
            if verified:
                print(f"⚠️ Recovery code used: user={username} (残り{len(user['recovery_codes'])}件)")

        if not verified:
            au.record_failure(username, ip)
            print(f"❌ Login failed (MFA code): user={username!r} ip={ip}")
            raise HTTPException(status_code=401, detail="認証コードが正しくありません")

        au.save_users(data)
        au.record_success(username)
        print(f"✅ Login success: {username}")
        return _session_response(user)

@app.post("/admin/api/auth/setup/start")
async def auth_setup_start(body: dict):
    """MFA登録: シークレットとQRコードを発行（まだ有効化はしない）"""
    payload = au.parse_token(str(body.get("setup_token", "")), "setup")
    if not payload:
        raise HTTPException(status_code=401, detail="確認の有効期限が切れました。最初からログインし直してください")
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, payload["u"])
        if not user or user.get("token_epoch", 1) != payload.get("e") or user.get("mfa_enabled"):
            raise HTTPException(status_code=401, detail="確認の有効期限が切れました。最初からログインし直してください")
        secret = au.new_totp_secret()
        user["pending_totp_secret"] = secret
        au.save_users(data)
    uri = au.totp_uri(secret, user["username"])
    return {"secret": secret, "otpauth_uri": uri, "qr_svg": au.qr_svg(uri)}

@app.post("/admin/api/auth/setup/confirm")
async def auth_setup_confirm(body: dict, request: Request):
    """MFA登録: アプリに表示された6桁コードで確認し、多要素認証を有効化する"""
    ip = _client_ip(request)
    payload = au.parse_token(str(body.get("setup_token", "")), "setup")
    if not payload:
        raise HTTPException(status_code=401, detail="確認の有効期限が切れました。最初からログインし直してください")
    username = payload["u"]
    _lock_guard(username, ip)
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        if (not user or user.get("token_epoch", 1) != payload.get("e")
                or user.get("mfa_enabled") or not user.get("pending_totp_secret")):
            raise HTTPException(status_code=401, detail="確認の有効期限が切れました。最初からログインし直してください")
        step = au.verify_totp(user["pending_totp_secret"], str(body.get("code", "")))
        if step is None:
            au.record_failure(username, ip)
            raise HTTPException(status_code=400, detail="認証コードが正しくありません。アプリに表示されている最新の6桁を入力してください")

        plain_codes, hashed_codes = au.generate_recovery_codes()
        user["totp_secret"] = user.pop("pending_totp_secret")
        user["pending_totp_secret"] = ""
        user["mfa_enabled"] = True
        user["totp_last_step"] = step
        user["recovery_codes"] = hashed_codes
        au.save_users(data)
        au.record_success(username)
        print(f"✅ MFA enabled: {username}")
        res = _session_response(user)
        res["recovery_codes"] = plain_codes   # 平文はこの1回のみ表示
        return res

@app.get("/admin/api/user")
async def get_current_user(username: str = Depends(verify_admin)):
    """ログイン中のユーザー情報"""
    user = au.find_user(au.load_users(), username)
    return {
        "username": username,
        "mfa_enabled": bool(user.get("mfa_enabled")),
        "recovery_codes_remaining": len(user.get("recovery_codes", [])),
        "must_change_password": bool(user.get("must_change_password")),
    }

@app.put("/admin/api/user/password")
async def change_password(body: dict, username: str = Depends(verify_admin)):
    """パスワード変更（本人）。変更後は全端末のログインが無効になるため、再ログインが必要。"""
    old_password = str(body.get("old_password", ""))
    new_password = str(body.get("new_password", ""))
    if not old_password or not new_password:
        raise HTTPException(status_code=400, detail="現在のパスワードと新しいパスワードを入力してください")
    problem = au.check_password_policy(new_password, username)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        if not au.verify_password_hash(user.get("password_hash", ""), old_password):
            raise HTTPException(status_code=400, detail="現在のパスワードが正しくありません")
        user["password_hash"] = au.hash_password(new_password)
        user["must_change_password"] = False
        user["token_epoch"] = user.get("token_epoch", 1) + 1
        au.save_users(data)
    print(f"✅ Password changed: {username}")
    return {"status": "ok", "message": "Password changed successfully"}

@app.post("/admin/api/user/recovery-codes/regenerate")
async def regenerate_recovery_codes(body: dict, username: str = Depends(verify_admin)):
    """リカバリーコードの再発行（現在の認証コードで本人確認。旧コードは全て無効になる）"""
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        step = au.verify_totp(user.get("totp_secret", ""), str(body.get("code", "")), user.get("totp_last_step", 0))
        if step is None:
            raise HTTPException(status_code=400, detail="認証コードが正しくありません")
        plain_codes, hashed_codes = au.generate_recovery_codes()
        user["recovery_codes"] = hashed_codes
        user["totp_last_step"] = step
        au.save_users(data)
    return {"status": "ok", "recovery_codes": plain_codes}

# ---- 管理者アカウント管理（管理者全員が操作可能） ----

@app.get("/admin/api/users")
async def list_admin_users(me: str = Depends(verify_admin)):
    users = au.load_users()["users"]
    return {"me": me, "users": [
        {
            "username": u["username"],
            "mfa_enabled": bool(u.get("mfa_enabled")),
            "must_change_password": bool(u.get("must_change_password")),
            "recovery_codes_remaining": len(u.get("recovery_codes", [])),
            "created_at": u.get("created_at", ""),
        } for u in users
    ]}

@app.post("/admin/api/users")
async def create_admin_user(body: dict, me: str = Depends(verify_admin)):
    """管理者を追加（初期パスワードを設定。本人は初回ログインでMFA登録とパスワード変更を行う）"""
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))
    if not username or len(username) > 50 or any(c in username for c in ' /\\:'):
        raise HTTPException(status_code=400, detail="ユーザー名は50文字以内で、空白や / \\ : は使えません")
    problem = au.check_password_policy(password, username)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    with au._users_lock:
        data = au.load_users()
        if au.find_user(data, username):
            raise HTTPException(status_code=400, detail=f"ユーザー '{username}' は既に存在します")
        data["users"].append(au.new_user_record(username, password, must_change_password=True))
        au.save_users(data)
    print(f"👤 Admin user created: {username} (by {me})")
    return {"status": "created", "username": username}

@app.delete("/admin/api/users/{username}")
async def delete_admin_user(username: str, me: str = Depends(verify_admin)):
    if username == me:
        raise HTTPException(status_code=400, detail="自分自身は削除できません")
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        if not user:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
        data["users"].remove(user)
        au.save_users(data)
    print(f"🗑️ Admin user deleted: {username} (by {me})")
    return {"status": "deleted", "username": username}

@app.post("/admin/api/users/{username}/reset-mfa")
async def reset_admin_mfa(username: str, me: str = Depends(verify_admin)):
    """スマホ紛失・機種変更時: 対象ユーザーのMFAを解除（次回ログイン時に再登録）。ログイン中の端末も無効化。"""
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        if not user:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
        user.update({"mfa_enabled": False, "totp_secret": "", "pending_totp_secret": "",
                     "totp_last_step": 0, "recovery_codes": [],
                     "token_epoch": user.get("token_epoch", 1) + 1})
        au.save_users(data)
    print(f"🔄 MFA reset: {username} (by {me})")
    return {"status": "ok", "username": username}

@app.post("/admin/api/users/{username}/reset-password")
async def reset_admin_password(username: str, body: dict, me: str = Depends(verify_admin)):
    """パスワード忘れ時: 一時パスワードを設定（本人は次回ログイン後に変更）。"""
    password = str(body.get("new_password", ""))
    problem = au.check_password_policy(password, username)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    with au._users_lock:
        data = au.load_users()
        user = au.find_user(data, username)
        if not user:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
        user["password_hash"] = au.hash_password(password)
        user["must_change_password"] = True
        user["token_epoch"] = user.get("token_epoch", 1) + 1
        au.save_users(data)
    print(f"🔄 Password reset: {username} (by {me})")
    return {"status": "ok", "username": username}

@app.get("/admin/users", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_users():
    """管理画面 - 管理者アカウント管理"""
    f = admin_path / "users.html"
    if not f.exists():
        return HTMLResponse("<h1>admin users.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/api/synonyms")
async def get_synonyms():
    """Synonyms.jsonã‚’è¿”ã™"""
    try:
        with open('synonyms.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Synonymsèª­ã¿è¾¼ã¿ã‚¨ãƒ©ãƒ¼: {e}")
        return {}


# ============================================
# 設定API
# ============================================

@app.get("/api/config")
async def get_config():
    """設定を取得"""
    print("📋 Config read requested")
    
    if not CONFIG_PATH.exists():
        # デフォルト設定を作成
        default_config = {
            "faq_search_enabled": True,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "semantic_weight": SEMANTIC_WEIGHT,
            "title_weight": TITLE_WEIGHT
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"✅ Created default config.json")
        except Exception as e:
            print(f"❌ Failed to create config.json: {e}")
        return default_config
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Config loaded: {config}")
        return config
    except Exception as e:
        print(f"❌ Config read error: {e}")
        return {
            "faq_search_enabled": True,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "semantic_weight": SEMANTIC_WEIGHT,
            "title_weight": TITLE_WEIGHT
        }


# ============================================================
# ファイル管理API (Persistent Disk)
# ============================================================

# カテゴリとディレクトリのマッピング
FILE_CATEGORIES = {
    "manuals":    FILES_MANUALS,
    "tools":      FILES_TOOLS,
    "installers": FILES_INSTALLERS,
    "other":      FILES_OTHER,
}

CATEGORY_LABELS = {
    "manuals":    "手順書・マニュアル",
    "tools":      "便利ツール",
    "installers": "インストーラー",
    "other":      "その他",
}

# ダウンロード時にインライン表示するMIMEタイプ（PDFなど）
INLINE_MIME_TYPES = {
    ".pdf": "application/pdf",
}

# ファイルのMIMEタイプマッピング
MIME_TYPES = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".doc":  "application/msword",
    ".xls":  "application/vnd.ms-excel",
    ".zip":  "application/zip",
    ".exe":  "application/octet-stream",
    ".msi":  "application/octet-stream",
    ".reg":  "application/octet-stream",
    ".txt":  "text/plain; charset=utf-8",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
}

def get_mime_type(filename: str) -> str:
    """ファイルのMIMEタイプを取得"""
    ext = pathlib.Path(filename).suffix.lower()
    return MIME_TYPES.get(ext, "application/octet-stream")

def get_file_category_dir(category: str) -> pathlib.Path:
    """カテゴリのディレクトリを取得"""
    if category not in FILE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    return FILE_CATEGORIES[category]

@app.get("/admin/api/files", dependencies=[Depends(verify_admin)])
async def list_files(category: str = "all"):
    """ファイル一覧を取得"""
    result = []

    categories = FILE_CATEGORIES if category == "all" else {category: FILE_CATEGORIES.get(category)}
    if category != "all" and category not in FILE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    for cat_key, cat_dir in categories.items():
        if not cat_dir or not cat_dir.exists():
            continue
        for f in sorted(cat_dir.iterdir()):
            if f.is_file():
                stat = f.stat()
                result.append({
                    "category":      cat_key,
                    "category_label": CATEGORY_LABELS.get(cat_key, cat_key),
                    "filename":      f.name,
                    "size":          stat.st_size,
                    "size_label":    _format_size(stat.st_size),
                    "modified":      datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "url":           f"/files/{cat_key}/{f.name}",
                    "mime_type":     get_mime_type(f.name),
                })

    return {"files": result, "total": len(result)}

@app.post("/admin/api/files/{category}", dependencies=[Depends(verify_admin)])
async def upload_file(category: str, file: UploadFile):
    """ファイルをアップロード"""
    cat_dir = get_file_category_dir(category)

    # ファイル名のサニタイズ（パストラバーサル対策）
    safe_name = pathlib.Path(file.filename).name
    if not safe_name or safe_name.startswith('.'):
        raise HTTPException(status_code=400, detail="Invalid filename")

    dest = cat_dir / safe_name

    # 既存ファイルのチェック
    if dest.exists():
        raise HTTPException(status_code=409, detail=f"File already exists: {safe_name}")

    # ファイルを保存
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MBずつ読み込み
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    stat = dest.stat()
    print(f"📤 File uploaded: {category}/{safe_name} ({_format_size(stat.st_size)})")

    return {
        "status":        "ok",
        "category":      category,
        "filename":      safe_name,
        "size":          stat.st_size,
        "size_label":    _format_size(stat.st_size),
        "url":           f"/files/{category}/{safe_name}",
    }

@app.delete("/admin/api/files/{category}/{filename}", dependencies=[Depends(verify_admin)])
async def delete_file(category: str, filename: str):
    """ファイルを削除"""
    cat_dir = get_file_category_dir(category)

    # パストラバーサル対策
    safe_name = pathlib.Path(filename).name
    target = cat_dir / safe_name

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    target.unlink()
    print(f"🗑️ File deleted: {category}/{safe_name}")
    return {"status": "ok", "deleted": safe_name}

@app.get("/files/{category}/{filename}")
async def download_file(category: str, filename: str):
    """ファイルをダウンロード（認証不要・FAQから参照用）"""
    if category not in FILE_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")

    # パストラバーサル対策
    safe_name = pathlib.Path(filename).name
    target = FILE_CATEGORIES[category] / safe_name

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    mime = get_mime_type(safe_name)
    ext  = pathlib.Path(safe_name).suffix.lower()

    # HTTPヘッダーはASCIIのみ許可されるため、日本語等の非ASCII文字を含む
    # ファイル名は必ずパーセントエンコードする（RFC 5987 / filename*）
    import urllib.parse
    encoded_name = urllib.parse.quote(safe_name)

    # PDFはブラウザでインライン表示、それ以外はダウンロード
    if ext in INLINE_MIME_TYPES:
        disposition = f"inline; filename*=UTF-8''{encoded_name}"
    else:
        disposition = f"attachment; filename*=UTF-8''{encoded_name}"

    def iter_file():
        with open(target, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=mime,
        headers={"Content-Disposition": disposition}
    )

def _format_size(size_bytes: int) -> str:
    """ファイルサイズを人間が読みやすい形式に変換"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.1f} GB"


@app.get("/admin/api/config", dependencies=[Depends(verify_admin)])
async def get_admin_config():
    """設定を取得（管理画面用）"""
    return await get_config()


@app.put("/admin/api/config", dependencies=[Depends(verify_admin)])
async def update_config(config_data: dict):
    """設定を更新"""
    print(f"⚙️ Updating config: {config_data}")
    
    try:
        # config.jsonに書き込み
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        # 書き込み確認
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        print(f"✅ Config saved successfully: {saved_config}")
        return {"status": "ok", "config": saved_config}
    except Exception as e:
        print(f"❌ Config update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

# ============================================
# 外部システム連携API (/api/v1/*)  ※ APIキー認証(X-API-Key)
# ============================================
# 管理画面用の /admin/api/* とは別に、パートナー企業や社内の
# 他システムから安全にFAQ・ログを参照/更新できるようにするAPI。

# ---- FAQ 参照・追加・更新 ----

@app.get("/api/v1/faq")
async def api_v1_list_faq(
    offset: int = 0,
    limit: int = 50,
    q: str = "",
    category: str = "",
    api_key_name: str = Depends(verify_api_key),
):
    """FAQ一覧取得（キーワード・カテゴリ絞り込み対応）"""
    await state.ensure_faq_loaded()
    items = state.faq_items_flat

    if category:
        items = [i for i in items if i.get("category", "") == category]
    if q:
        q_lower = q.lower()
        items = [
            i for i in items
            if q_lower in i.get("question", "").lower()
            or q_lower in i.get("answer", "").lower()
            or any(q_lower in kw.lower() for kw in i.get("keywords", []))
        ]

    total = len(items)
    page_items = items[offset:offset + limit]
    return {"items": page_items, "total": total, "has_more": (offset + limit) < total}

@app.get("/api/v1/faq/{faq_id}")
async def api_v1_get_faq(faq_id: str, api_key_name: str = Depends(verify_api_key)):
    """FAQ単体参照（一覧APIと同じ正規化済みフィールド構成で返す）"""
    await state.ensure_faq_loaded()
    item = next((i for i in state.faq_items_flat if i.get("id") == faq_id), None)
    if not item:
        raise HTTPException(404, f"FAQ item '{faq_id}' not found")
    return item

@app.post("/api/v1/faq")
async def api_v1_create_faq(item: dict, background_tasks: BackgroundTasks, api_key_name: str = Depends(verify_api_key)):
    """FAQ新規追加（外部システムから）"""
    faq_id = await _create_faq_item_internal(item, changed_by=api_key_name)
    background_tasks.add_task(reload_faq_data)
    return {"status": "created", "id": faq_id}

@app.put("/api/v1/faq/{faq_id}")
async def api_v1_update_faq(faq_id: str, item: dict, background_tasks: BackgroundTasks, api_key_name: str = Depends(verify_api_key)):
    """FAQ更新（外部システムから）"""
    await _update_faq_item_internal(faq_id, item, changed_by=api_key_name)
    background_tasks.add_task(reload_faq_data)
    return {"status": "updated", "id": faq_id}

@app.get("/api/v1/faq/{faq_id}/history")
async def api_v1_get_faq_history(faq_id: str, api_key_name: str = Depends(verify_api_key)):
    """FAQ変更履歴の参照（作成・更新・削除の記録）"""
    return {"faq_id": faq_id, "history": get_faq_history(faq_id)}

# ---- 検索・クリックログ参照 ----

@app.get("/api/v1/logs")
async def api_v1_get_logs(
    type: str = Query("", description="faq または video で絞り込み。空の場合は全件"),
    date_from: str = Query("", description="YYYY-MM-DD形式。指定日以降のログのみ"),
    date_to: str = Query("", description="YYYY-MM-DD形式。指定日以前のログのみ"),
    offset: int = 0,
    limit: int = 100,
    api_key_name: str = Depends(verify_api_key),
):
    """検索・クリックの生ログを参照する"""
    rows = parse_logs()

    if type:
        rows = [r for r in rows if r.get("result_type") == type]
    if date_from:
        try:
            df = datetime.strptime(date_from, "%Y-%m-%d")
            rows = [r for r in rows if r["dt"].replace(tzinfo=None) >= df]
        except ValueError:
            raise HTTPException(400, "date_from must be in YYYY-MM-DD format")
    if date_to:
        try:
            dt_to = datetime.strptime(date_to, "%Y-%m-%d")
            rows = [r for r in rows if r["dt"].replace(tzinfo=None) <= dt_to]
        except ValueError:
            raise HTTPException(400, "date_to must be in YYYY-MM-DD format")

    total = len(rows)
    # 新しい順に並べ替えてページング
    rows_sorted = sorted(rows, key=lambda r: r["dt"], reverse=True)
    page_rows = rows_sorted[offset:offset + limit]

    logs = [
        {
            "timestamp": r["dt"].isoformat(),
            "type": r.get("result_type", ""),
            "query": r.get("query", ""),
            "result_id": r.get("result_id", ""),
        }
        for r in page_rows
    ]
    return {"logs": logs, "total": total, "has_more": (offset + limit) < total}

@app.get("/api/v1/logs/summary")
async def api_v1_logs_summary(month: str = Query(..., description="YYYY-MM形式"), api_key_name: str = Depends(verify_api_key)):
    """月別サマリー（日別件数・FAQ/動画別トップキーワード）"""
    return await get_log_summary(month)

@app.get("/api/v1/ranking/faq")
async def api_v1_ranking_faq(limit: int = 10, api_key_name: str = Depends(verify_api_key)):
    """FAQクリックランキング"""
    return await get_faq_ranking(limit)

@app.get("/api/v1/ranking/video")
async def api_v1_ranking_video(limit: int = 10, api_key_name: str = Depends(verify_api_key)):
    """動画クリックランキング"""
    return await get_video_ranking(limit)

# ---- APIキー管理（管理画面のBasic認証で保護） ----

@app.get("/admin/api/apikeys", dependencies=[Depends(verify_admin)])
async def admin_list_api_keys():
    """APIキー一覧（キー本体はマスクして返す）"""
    keys = load_api_keys()
    masked = [
        {
            "name": k.get("name", ""),
            "enabled": k.get("enabled", True),
            "created_at": k.get("created_at", ""),
            "key_preview": (k.get("key", "")[:6] + "..." + k.get("key", "")[-4:]) if len(k.get("key", "")) > 10 else "****",
        }
        for k in keys
    ]
    return {"keys": masked}

@app.post("/admin/api/apikeys", dependencies=[Depends(verify_admin)])
async def admin_create_api_key(payload: dict):
    """
    新規APIキーを発行する。
    payload例: {"name": "グラフテック連携システム"}
    発行したキーの平文はこのレスポンスでのみ返却される（以降はマスク表示のみ）。
    """
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name is required")

    keys = load_api_keys()
    if any(k.get("name") == name for k in keys):
        raise HTTPException(400, f"API key with name '{name}' already exists")

    new_key = secrets.token_urlsafe(32)
    keys.append({
        "key": new_key,
        "name": name,
        "enabled": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    save_api_keys(keys)

    return {"status": "created", "name": name, "key": new_key}

@app.delete("/admin/api/apikeys/{name}", dependencies=[Depends(verify_admin)])
async def admin_delete_api_key(name: str):
    """APIキーを削除する"""
    keys = load_api_keys()
    new_keys = [k for k in keys if k.get("name") != name]
    if len(new_keys) == len(keys):
        raise HTTPException(404, f"API key '{name}' not found")
    save_api_keys(new_keys)
    return {"status": "deleted", "name": name}

@app.patch("/admin/api/apikeys/{name}", dependencies=[Depends(verify_admin)])
async def admin_toggle_api_key(name: str, payload: dict):
    """APIキーの有効/無効を切り替える。payload例: {"enabled": false}"""
    keys = load_api_keys()
    found = False
    for k in keys:
        if k.get("name") == name:
            k["enabled"] = bool(payload.get("enabled", True))
            found = True
            break
    if not found:
        raise HTTPException(404, f"API key '{name}' not found")
    save_api_keys(keys)
    return {"status": "updated", "name": name}


if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.getenv("PORT", 8000))
        print(f"🚀 Starting server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise
