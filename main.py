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
from datetime import datetime, timezone, timedelta
import threading
import asyncio
import ja_text  # 日本語解析（形態素解析）。未インストールでも起動できる作り（ja_text.py参照）
import io
import time
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

# --- 埋め込みモデルごとの前処理 ---
# multilingual-e5 系は「検索語の先頭に query: 」「文書の先頭に passage: 」を付ける前提で学習されているため、
# 付けないと精度が大きく落ちる。現行モデル(MiniLM)は付けない（＝従来と完全に同じ動き）。
# 他のモデルで必要な場合は環境変数 EMBEDDING_QUERY_PREFIX / EMBEDDING_PASSAGE_PREFIX で上書きできる。
_IS_E5 = "e5" in EMBEDDING_MODEL.lower().rsplit("/", 1)[-1]
QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: " if _IS_E5 else "")
PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: " if _IS_E5 else "")

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
SEARCH_EVENTS_PATH = DATA_DIR / "search_events.jsonl"  # 検索実行ログ（追記専用）
EVAL_CASES_PATH = DATA_DIR / "eval_cases.json"        # 精度計測: 手動で登録したテスト問題
EVAL_HISTORY_PATH = DATA_DIR / "eval_history.json"    # 精度計測: 計測結果の履歴

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
# モデルごとにキャッシュファイルを分ける（モデルを切り替えて戻した時に再計算不要にするため）。
# 現行モデルは従来どおりのファイル名（既存キャッシュをそのまま使う）。
_MODEL_SUFFIX = "" if EMBEDDING_MODEL == DEFAULT_MODEL_NAME else "__" + re.sub(r"[^A-Za-z0-9_.-]+", "_", EMBEDDING_MODEL)
VIDEO_EMBEDDINGS_CACHE_PATH = CACHE_DIR / f"video_embeddings{_MODEL_SUFFIX}.npy"
VIDEO_CHUNK_META_CACHE_PATH = CACHE_DIR / f"video_chunks_meta{_MODEL_SUFFIX}.json"
# 区切り（チャンク）ごとの計算結果キャッシュ。文章が同じ区切りは再計算しない。
# 動画を1本更新しても、変わった区切りだけを計算すれば済むようにするため。
VIDEO_CHUNK_VEC_CACHE_PATH = CACHE_DIR / f"video_chunk_vectors{_MODEL_SUFFIX}.npy"
VIDEO_CHUNK_KEY_CACHE_PATH = CACHE_DIR / f"video_chunk_keys{_MODEL_SUFFIX}.json"
# 埋め込み計算の設定
#  ENCODE_BATCH_SIZE: 一度に計算する件数。大きいほど速いがメモリを使う。e5系は長い文章(最大512トークン)を
#                     扱うため、64件まとめるとメモリが1GB以上一時的に増えることがあり、小さめにする。
#  ENCODE_CHECKPOINT: この件数ごとに途中経過を保存（再起動しても続きから再開できるように）。
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "16"))
ENCODE_CHECKPOINT = int(os.getenv("ENCODE_CHECKPOINT", "256"))
# 動画検索で、準備（初回計算）が終わるまで待つ最大秒数。超えたら「準備中」として空の結果を返す。
VIDEO_PREPARE_WAIT_SECONDS = float(os.getenv("VIDEO_PREPARE_WAIT_SECONDS", "20"))

# 埋め込み計算の排他。検索用データの作り直し（別スレッド）とお客様の検索（本体）が同じモデルを
# 同時に使わないようにする。作り直しは小分け（ENCODE_BATCH_SIZE件ずつ）でロックを取るので、
# お客様の検索が待つのは長くても小分け1回分（数秒以内）で済む。
_encode_lock = threading.Lock()

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
        self.model_load_seconds: Optional[float] = None
        # 動画データの再計算要求（インポート・削除・編集のたびにTrue）。計算中に要求が来ても取りこぼさないため
        self.video_dirty = False
        self.video_loading = False
        # 検索が使う動画データ一式（動画一覧・区切り→動画の対応・索引）を1つにまとめたもの。
        # 再計算中に別スレッドで差し替わっても、検索が「新旧が混ざった状態」を見ないようにするため
        self.video_snapshot = ([], [], None)
        self.video_progress: Dict[str, Any] = {}
        
    # ------------------------------------------------------------
    # 読み込み処理（モデル・動画・FAQ）
    # なぜ別スレッドか: モデルの読み込みや埋め込み計算は数十秒〜数分かかる（1CPU・モデル切替直後）。
    # 本体の処理（イベントループ）で実行すると、その間ヘルスチェックや他のお客様の検索まで止まるため、
    # 別スレッドで実行し、待っている検索は完了を待つだけにする。同時に2回読み込まないようロックする。
    # ------------------------------------------------------------
    _load_lock = threading.RLock()

    def _load_model_sync(self):
        with self._load_lock:
            if not self.model_loaded:
                t0 = time.time()
                print(f"🔄 Loading model: {EMBEDDING_MODEL} (query prefix={QUERY_PREFIX!r}, passage prefix={PASSAGE_PREFIX!r})")
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self.model_loaded = True
                self.model_load_seconds = round(time.time() - t0, 1)
                print(f"✅ Model loaded ({self.model_load_seconds}s)")

    async def ensure_model_loaded(self):
        if not self.model_loaded:
            await asyncio.to_thread(self._load_model_sync)

    def _load_video_sync(self):
        with self._load_lock:
            self._load_video_locked()

    async def ensure_video_loaded(self):
        if not self.video_loaded:
            await asyncio.to_thread(self._load_video_sync)

    def _load_faq_sync(self):
        with self._load_lock:
            self._load_faq_locked()

    async def ensure_faq_loaded(self):
        if not self.faq_loaded:
            await asyncio.to_thread(self._load_faq_sync)

    def _load_video_locked(self):
        """動画データの読み込みと検索用データの作成 ※ロック取得済みで呼ぶこと

        ・区切りごとの計算結果をキャッシュし、文章が変わった区切りだけを計算する
        ・計算の途中経過も保存するので、再起動しても続きから再開できる
        ・計算中も、検索は直前のデータ（video_snapshot）で続けられる。完成したら一度に差し替える
        ・計算中に新しい変更が入った場合（video_dirty）は、終わった後にもう一度作り直す
        """
        if self.video_loaded and not self.video_dirty:
            return
        self._load_model_sync()
        self.video_loading = True
        try:
            while True:
                self.video_dirty = False
                self._build_video_index()
                if not self.video_dirty:
                    break
                print("🔁 Video data changed during rebuild; rebuilding again with the latest data")
        finally:
            self.video_loading = False

    def _build_video_index(self):
        t0 = time.time()
        print("🔄 Loading video data...")
        videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                videos = json.load(f)
        if SYNONYMS_PATH.exists():
            try:
                with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to read synonyms (keeping previous): {e}")

        # ------------------------------------------------------------
        # コーパス構築（チャンク分割対応）
        # transcriptが長い動画でも全体を検索対象にするため、タイトル+説明文+transcriptチャンクの単位で
        # 複数ベクトルを生成する。同じ動画の複数チャンクは chunk_map で動画インデックスに紐付け、
        # 検索時にスコアを集約する。
        # ------------------------------------------------------------
        corpus: List[str] = []
        chunk_map: List[int] = []
        for v_idx, v in enumerate(videos):
            title = v.get("title", "") or ""
            description = v.get("description", "") or ""
            transcript = v.get("transcript", "") or ""
            base = f"{title} {description}".strip()
            if transcript:
                text_chunks = chunk_text(transcript) or [""]
                for tc in text_chunks:
                    corpus.append(normalize_text(f"{base} {tc}".strip()))
                    chunk_map.append(v_idx)
            else:
                corpus.append(normalize_text(base))
                chunk_map.append(v_idx)

        embeddings = self._encode_with_chunk_cache(corpus) if corpus else None
        index = build_optimized_index(embeddings) if embeddings is not None and len(embeddings) > 0 else None

        # まとめて差し替え（検索は video_snapshot を1回読むだけなので、新旧が混ざらない）
        self.video_snapshot = (videos, chunk_map, index)
        self.videos = videos
        self.text_corpus = corpus
        self.video_chunk_map = chunk_map
        self.video_embeddings = embeddings
        self.video_index = index
        self.video_loaded = True
        print(f"✅ Video data loaded: {len(videos)} videos, {len(corpus)} chunks ({time.time() - t0:.1f}s)")

    # ---------------- 区切りごとの計算結果キャッシュ ----------------
    def _chunk_key(self, text: str) -> str:
        return hashlib.sha1((PASSAGE_PREFIX + "\x1f" + text).encode("utf-8")).hexdigest()

    def _load_chunk_cache(self) -> Dict[str, np.ndarray]:
        cache: Dict[str, np.ndarray] = {}
        try:
            if VIDEO_CHUNK_VEC_CACHE_PATH.exists() and VIDEO_CHUNK_KEY_CACHE_PATH.exists():
                with open(VIDEO_CHUNK_KEY_CACHE_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                vecs = np.load(VIDEO_CHUNK_VEC_CACHE_PATH)
                keys = meta.get("keys", [])
                if meta.get("model") == EMBEDDING_MODEL and len(keys) == len(vecs):
                    cache = {k: vecs[i] for i, k in enumerate(keys)}
                else:
                    print("♻️ Chunk cache ignored (model changed or file mismatch)")
        except Exception as e:
            print(f"⚠️ Failed to read chunk cache (will recompute): {e}")
            cache = {}
        # 旧形式のキャッシュ（data.json全体で1つ）があれば取り込む。今回の更新で全件の再計算が起きないように
        if VIDEO_EMBEDDINGS_CACHE_PATH.exists() and VIDEO_CHUNK_META_CACHE_PATH.exists():
            try:
                with open(VIDEO_CHUNK_META_CACHE_PATH, "r", encoding="utf-8") as f:
                    old = json.load(f)
                if (old.get("model") == EMBEDDING_MODEL and old.get("passage_prefix", "") == PASSAGE_PREFIX):
                    old_vecs = np.load(VIDEO_EMBEDDINGS_CACHE_PATH)
                    old_corpus = old.get("text_corpus", [])
                    if len(old_corpus) == len(old_vecs):
                        n = 0
                        for t, vec in zip(old_corpus, old_vecs):
                            k = self._chunk_key(t)
                            if k not in cache:
                                cache[k] = vec
                                n += 1
                        if n:
                            print(f"📦 Imported {n} vectors from legacy video cache")
            except Exception as e:
                print(f"⚠️ Failed to import legacy video cache: {e}")
        return cache

    def _save_chunk_cache(self, cache: Dict[str, np.ndarray], keep: Optional[set] = None) -> None:
        """キャッシュを保存（一時ファイルに書いてから置き換えるので、途中で落ちても壊れない）"""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            keys = [k for k in cache if keep is None or k in keep]
            if not keys:
                return
            vecs = np.stack([cache[k] for k in keys]).astype("float32")
            tmp_v = VIDEO_CHUNK_VEC_CACHE_PATH.with_suffix(".tmp.npy")
            tmp_k = VIDEO_CHUNK_KEY_CACHE_PATH.with_suffix(".tmp.json")
            np.save(tmp_v, vecs)
            with open(tmp_k, "w", encoding="utf-8") as f:
                json.dump({"model": EMBEDDING_MODEL, "passage_prefix": PASSAGE_PREFIX, "keys": keys,
                           "updated_at": datetime.now(timezone.utc).isoformat()}, f)
            # ベクトル→キーの順に置き換え。間で落ちても件数不一致として読み捨てられ、再計算になるだけ
            os.replace(tmp_v, VIDEO_CHUNK_VEC_CACHE_PATH)
            os.replace(tmp_k, VIDEO_CHUNK_KEY_CACHE_PATH)
        except Exception as e:
            print(f"⚠️ Failed to save chunk cache: {e}")

    def _encode_with_chunk_cache(self, corpus: List[str]) -> np.ndarray:
        cache = self._load_chunk_cache()
        keys = [self._chunk_key(t) for t in corpus]
        missing: Dict[str, str] = {}
        for k, t in zip(keys, corpus):
            if k not in cache and k not in missing:
                missing[k] = t
        total = len(missing)
        print(f"🧮 Video chunks: {len(corpus)} total, {sum(1 for k in keys if k in cache)} from cache, {total} unique to compute")
        if total:
            items = sorted(missing.items(), key=lambda kv: len(kv[1]))
            done = 0
            t0 = time.time()
            for start in range(0, total, ENCODE_CHECKPOINT):
                block = items[start:start + ENCODE_CHECKPOINT]
                vecs = encode_passages([t for _, t in block])
                faiss.normalize_L2(vecs)
                for (k, _), vec in zip(block, vecs):
                    cache[k] = vec
                done += len(block)
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                self.video_progress = {"done": done, "total": total, "eta_seconds": round(eta)}
                print(f"   … {done}/{total} chunks computed ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")
                self._save_chunk_cache(cache)          # 途中経過の保存（再起動しても続きから）
        # 今のデータに無い区切りは削除して保存（ファイルが際限なく大きくならないように）
        self._save_chunk_cache(cache, keep=set(keys))
        self.video_progress = {}
        return np.stack([cache[k] for k in keys]).astype("float32")

    def _load_faq_locked(self):
        # FAQデータの遅延ロード ※ロック取得済みで呼ぶこと
        if not self.faq_loaded:
            self._load_model_sync()
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
                self.faq_embeddings = encode_passages(self.faq_corpus)
                faiss.normalize_L2(self.faq_embeddings)
                self.faq_index = build_optimized_index(self.faq_embeddings)
            
            self.faq_loaded = True
            print(f"âœ… FAQ data loaded: {len(self.faq_items_flat)} items")

state = AppState()

# ============================================
# ãƒ¦ãƒ¼ãƒ†ã‚£ãƒªãƒ†ã‚£é–¢æ•°
# ============================================

def encode_passages(texts: List[str]) -> np.ndarray:
    """文書側（動画の区切り・FAQ）の埋め込み計算。小分けにしてロックを取り、検索を長く待たせない。
    e5系の場合は先頭に "passage: " を付ける（PASSAGE_PREFIX）。"""
    out = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = [PASSAGE_PREFIX + t for t in texts[i:i + ENCODE_BATCH_SIZE]]
        with _encode_lock:
            out.append(state.model.encode(batch, show_progress_bar=False, convert_to_numpy=True,
                                          batch_size=ENCODE_BATCH_SIZE))
    return np.vstack(out).astype("float32") if out else np.zeros((0, 1), dtype="float32")


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
    
    # 同義語展開（両方向）
    #  ・検索語に「登録語」が含まれる → その同義語を追加（例: プリンタ → プリンター）
    #  ・検索語に「同義語」が含まれる → 登録語を追加（例: プリンター → プリンタ）
    # 以前はお客様画面(JavaScript)でも同じ展開をしていたが、辞書の読み込みに失敗していたため、
    # サーバー側に一本化した。展開後の文字列は「意味検索」にだけ使い、
    # キーワード一致・完全一致の判定には元の検索語を使う（同義語が混ざると完全一致が効かなくなるため）。
    q_lower = query.lower()
    for term, syns in lang_synonyms.items():
        if not isinstance(syns, list):
            continue
        if term and term.lower() in q_lower:
            for s_ in syns:
                if s_ and s_ not in expanded_terms:
                    expanded_terms.append(s_)
        for s_ in syns:
            if s_ and s_.lower() in q_lower and term not in expanded_terms:
                expanded_terms.append(term)
    
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
def keyword_match_score(query: str, text: str, use_ja: bool = False) -> float:
    """キーワード一致度（0.0〜1.0）。
    use_ja=False: 従来の title_match_score と完全に同じ。
    use_ja=True : 日本語解析（単語単位の一致率）を主に、文字bi-gramを補助に使う。
                  例「プリンタ 印刷できない」→ 語「プリンター」「印刷」がいくつ含まれるか。
                  bi-gramだけだと「印刷」と無関係な「刷で」等の一致でも点が入るため、それを抑える。"""
    base = title_match_score(query, text)
    if not use_ja or base >= 1.0 or not query or not text:
        return base
    cov = ja_text.token_coverage(query, text)
    if cov is None:          # 内容語が取れない・解析不可 → 従来どおり
        return base
    q = normalize_text(query)
    t = normalize_text(text)
    words = q.split()
    word_score = (sum(1 for w in words if w and w in t) / len(words)) if len(words) > 1 else 0.0
    bigrams = _char_bigrams(q.replace(" ", ""))
    bigram_score = (sum(1 for bg in bigrams if bg in t) / len(bigrams)) if bigrams else 0.0
    return max(word_score, 0.7 * cov + 0.3 * bigram_score)


def faq_keyword_match_score(query: str, item: Dict[str, Any], use_ja: bool = False) -> float:
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
    return keyword_match_score(query, combined_text, use_ja)




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
    # 検索モデル・データを裏で先に読み込む（最初のお客様が数十秒〜数分待たされるのを防ぐ。
    # モデル切り替え直後は埋め込みの再計算に数分かかるため特に重要）。
    # 別スレッドで動くので、起動完了・ヘルスチェックは待たせない。WARMUP_ON_START=0 で無効化できる。
    reload_synonyms()
    if os.getenv("WARMUP_ON_START", "1") != "0":
        async def _warmup():
            try:
                await state.ensure_faq_loaded()
                await state.ensure_video_loaded()
                print(f"🔥 Warmup finished (memory {_rss_mb()} MB)")
            except Exception as e:
                print(f"⚠️ Warmup failed (will retry on first search): {e}")
        app.state.warmup_task = asyncio.create_task(_warmup())
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
        "video_items_count": len(state.videos),
    }

# ============================================
# å‹•ç”»æ¤œç´¢ã‚¨ãƒ³ãƒ‰ãƒã‚¤ãƒ³ãƒˆ
# ============================================

def video_search_core(query: str, language: Optional[str], config: Dict[str, Any], offset_limit: int = MAX_PAGE_LIMIT,
                      include_rejected: bool = False, verbose: bool = True) -> List[Dict[str, Any]]:
    """動画検索の本体（並び順まで確定した全件を返す）。/search と精度計測の両方から使う。
    include_rejected=True の時は、しきい値で落ちた動画も "_passed": False を付けて返す（精度計測のしきい値分析用）。"""
    normalized_query = normalize_text(query)
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")

    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    use_ja = bool(config.get("ja_tokenizer_enabled", False)) and ja_text.available()
    if verbose:
        print(f"🔍 Video search: query='{query}', threshold={threshold}, config={config}")

    with _encode_lock:
        query_embedding = state.model.encode([QUERY_PREFIX + expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # チャンク単位で検索するため、動画数ではなく総チャンク数を基準に取得件数を決定。
    # 同一動画の複数チャンクがヒットしても動画単位に集約するため、多めに取得する。
    videos, chunk_map, video_index = state.video_snapshot   # 再計算中の差し替えと混ざらないよう、1回だけ読む
    total_chunks = len(chunk_map)
    if video_index is None or total_chunks == 0:
        return []
    k = min(max((offset_limit + 100) * 10, 500), total_chunks)
    distances, indices = video_index.search(query_embedding, k)

    semantic_weight = config.get("semantic_weight", SEMANTIC_WEIGHT)
    title_weight = config.get("title_weight", TITLE_WEIGHT)

    # チャンク単位のスコアを動画単位に集約（同じ動画は最も高いチャンクのスコアを採用）
    best_score_per_video = {}
    all_scores = []
    for chunk_idx, score in zip(indices[0], distances[0]):
        if 0 <= chunk_idx < len(chunk_map):
            all_scores.append(float(score))
            v_idx = chunk_map[chunk_idx]
            s_ = float(score)
            if v_idx not in best_score_per_video or s_ > best_score_per_video[v_idx]:
                best_score_per_video[v_idx] = s_

    results = []
    for v_idx, semantic_score in best_score_per_video.items():
        if not (0 <= v_idx < len(videos)):
            continue
        video = videos[v_idx].copy()
        video_lang = video.get("language", "ja")
        if language != "all" and video_lang != language:
            continue
        title_bonus = keyword_match_score(query, video.get("title", ""), use_ja)
        hybrid_score = semantic_score * semantic_weight + title_bonus * title_weight
        video["score"] = hybrid_score
        video["semantic_score"] = semantic_score
        video["title_score"] = title_bonus
        passed = semantic_score >= threshold          # 閾値チェック（セマンティックスコアで判定）
        if passed:
            results.append(video)
        elif include_rejected:
            video["_passed"] = False
            results.append(video)

    if verbose and all_scores:
        print(f"  Total candidate chunks: {len(all_scores)} (from {total_chunks} total chunks)")
        print(f"  Unique videos matched: {len(best_score_per_video)}")
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Top 5 scores: {[f'{x:.3f}' for x in sorted(all_scores, reverse=True)[:5]]}")
        print(f"  Passed threshold ({threshold}): {sum(1 for r in results if r.get('_passed', True))} items")

    # タイトル完全一致を最優先、次点でハイブリッドスコアでソート
    query_norm = normalize_text(query)
    results.sort(key=lambda v: (
        -int(query_norm in normalize_text(v.get("title", ""))),
        -v["score"]
    ))
    return results


@app.get("/search")
async def search_videos(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0),
    language: str = Query(None)
):
    """動画検索（ページング対応）"""
    if not state.video_loaded:
        # 起動直後など、まだ検索用データが無い場合。計算に時間がかかることがあるので、
        # 一定時間だけ待ち、間に合わなければ「準備中」として空の結果を返す（お客様を長時間待たせない）
        task = getattr(state, "_video_prepare_task", None)
        if task is None or task.done():
            task = asyncio.ensure_future(state.ensure_video_loaded())
            state._video_prepare_task = task
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=VIDEO_PREPARE_WAIT_SECONDS)
        except asyncio.TimeoutError:
            print(f"⏳ Video search requested while preparing (progress: {state.video_progress})")
            return {"items": [], "has_more": False, "total_visible": 0, "preparing": True}

    if not state.video_snapshot[2]:
        return {"items": [], "has_more": False, "total_visible": 0}

    config = await get_config()
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    results = video_search_core(query, language, config, offset_limit=offset + limit)

    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total

    if results:
        scores = [r["score"] for r in results[:10]]
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
    
    config = await get_config()
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    results = faq_search_core(query, language, config, offset_limit=offset + limit)

    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total

    if results:
        scores = [r["score"] for r in results[:10]]
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


def faq_search_core(query: str, language: Optional[str], config: Dict[str, Any], offset_limit: int = MAX_PAGE_LIMIT,
                    include_rejected: bool = False, verbose: bool = True) -> List[Dict[str, Any]]:
    """FAQ検索の本体（意味検索＋キーワード一致。並び順まで確定した全件）。/faq/search と精度計測から使う。"""
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    use_ja = bool(config.get("ja_tokenizer_enabled", False)) and ja_text.available()
    if verbose:
        print(f"🔍 FAQ search: query='{query}', threshold={threshold}, config={config}")

    normalized_query = normalize_text(query)
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
    with _encode_lock:
        query_embedding = state.model.encode([QUERY_PREFIX + expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    k = min(offset_limit + 100, len(state.faq_items_flat))
    distances, indices = state.faq_index.search(query_embedding, k)

    semantic_weight = config.get("semantic_weight", SEMANTIC_WEIGHT)
    title_weight = config.get("title_weight", TITLE_WEIGHT)

    results = []
    all_scores = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.faq_items_flat):
            all_scores.append(float(score))
            item = state.faq_items_flat[idx].copy()
            item_lang = item.get("language", "ja")
            if language != "all" and item_lang != language:
                continue
            # 質問・回答・手順・キーワード等の全文一致度で、意味検索が拾いきれない専門用語を補完
            semantic_score = float(score)
            keyword_bonus = faq_keyword_match_score(query, item, use_ja)
            item["score"] = semantic_score * semantic_weight + keyword_bonus * title_weight
            item["semantic_score"] = semantic_score
            item["title_score"] = keyword_bonus
            # 意味検索のスコアが閾値以上、またはキーワード一致度が高ければ結果に含める
            passed = semantic_score >= threshold or keyword_bonus >= FAQ_KEYWORD_MATCH_MIN_SCORE
            if passed:
                results.append(item)
            elif include_rejected:
                item["_passed"] = False
                results.append(item)

    if verbose and all_scores:
        print(f"  Total candidates: {len(all_scores)}")
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Top 5 scores: {[f'{x:.3f}' for x in sorted(all_scores, reverse=True)[:5]]}")
        print(f"  Passed threshold ({threshold}): {sum(1 for r in results if r.get('_passed', True))} items")

    # 質問文完全一致を最優先、次点でハイブリッドスコアでソート
    query_norm = normalize_text(query)
    results.sort(key=lambda v: (
        -int(query_norm in normalize_text(v.get("question", ""))),
        -v["score"]
    ))
    return results

# ============================================
# ç®¡ç†API - ãƒ‡ãƒ¼ã‚¿ç·¨é›†
# ============================================

def reload_synonyms() -> None:
    """同義語辞書をファイルから読み直し、検索に即反映する（動画・FAQの再計算は不要）"""
    try:
        if SYNONYMS_PATH.exists():
            with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                state.synonyms = json.load(f)
        else:
            state.synonyms = {}
        print("🔁 Synonyms reloaded")
    except Exception as e:
        # 読み直しに失敗しても、直前の辞書のまま検索は続ける
        print(f"⚠️ Failed to reload synonyms (keeping previous): {e}")


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
    reload_synonyms()
    print(f"✅ Synonyms saved successfully")
    return {"status": "ok", "count": len(data)}

async def reload_video_data():
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ã®ãƒªãƒ­ãƒ¼ãƒ‰"""
    # 「作り直してほしい」という印だけ付けて、別スレッドで作り直す。
    # 作り直しの間も、検索は直前のデータで続ける（以前は完了まで検索を止めていたため、
    # 計算に時間がかかると検索全体が長時間待たされていた）。
    state.video_dirty = True
    try:
        await asyncio.to_thread(state._load_video_sync)
    except Exception as e:
        print(f"❌ Video reload failed (search keeps using previous data): {e}")

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
    reload_synonyms()
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
    reload_synonyms()
    
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
    reload_synonyms()
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
    
    # 1セルずつ読み直し、数式扱いされる値を無害化してから出力する（xl_safe参照）
    try:
        with open(SEARCH_LOG_PATH, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"❌ Failed to read logs for export: {e}")
        raise HTTPException(500, "ログファイルの読み込みに失敗しました")
    buf = io.StringIO()
    csv.writer(buf).writerows([[xl_safe(c) for c in row] for row in rows])
    csv_data = buf.getvalue()
    
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
# 検索実行ログ（検索の質の分析用）
# ============================================
# 既存のクリックログ(search_logs.csv)は「クリックされた時だけ」の記録のため、
# 「ヒットなしだった検索」「結果が出たのにクリックされなかった検索」が分かりません。
# そこで検索が落ち着いた時点（入力が止まった / Enter / 結果クリック前）で1件記録します。
#
# 形式: search_events.jsonl（1行1イベントの追記専用。書き込み途中で壊れても他の行に影響しない）
#   検索:   {"t": ISO日時(UTC), "sid": 検索ID, "q": キーワード, "f": FAQヒット数|null, "v": 動画ヒット数, "lang": "ja"}
#   クリック: {"t": ISO日時(UTC), "sid": 検索ID, "c": "faq"|"video"}
# 検索IDで検索とクリックを結び付けて集計する。
# 注意: 個人情報の入力を避ける注意書きは画面側にあるが、キーワードは200文字までに切り詰めて保存する。

JST = timezone(timedelta(hours=9))   # 日本時間（夏時間なし）
_events_lock = threading.Lock()
_event_rate: Dict[str, List[float]] = {}
EVENT_RATE_LIMIT_PER_MIN = 120       # 同一IPからの記録上限（悪意ある大量送信でファイルが膨らむのを防ぐ）
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,40}$")

def _event_rate_ok(ip: str) -> bool:
    now = time.time()
    hits = [t for t in _event_rate.get(ip, []) if now - t < 60]
    if len(hits) >= EVENT_RATE_LIMIT_PER_MIN:
        _event_rate[ip] = hits
        return False
    hits.append(now)
    _event_rate[ip] = hits
    if len(_event_rate) > 5000:   # メモリ肥大の防止
        _event_rate.clear()
    return True

def _append_event(obj: dict) -> None:
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    with _events_lock:
        with open(SEARCH_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def _to_count(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return max(0, min(int(v), 100000))
    except Exception:
        return None

@app.post("/api/search_event")
async def log_search_event(body: dict, request: Request):
    """検索実行の記録（お客様画面から、検索が落ち着いた時に1回だけ呼ばれる）"""
    try:
        sid = str(body.get("sid") or "")
        query = re.sub(r"\s+", " ", str(body.get("query") or "")).strip()[:200]
        ip = request.client.host if request.client else "unknown"
        if not _SID_RE.match(sid) or not query:
            return {"status": "ignored"}
        if not _event_rate_ok(ip):
            return {"status": "rate_limited"}
        _append_event({
            "t": datetime.now(timezone.utc).isoformat(),
            "sid": sid, "q": query,
            "f": _to_count(body.get("faq_count")),     # FAQ検索が無効の時はnull
            "v": _to_count(body.get("video_count")),
            "lang": str(body.get("language") or "")[:5],
        })
    except Exception as e:
        print(f"⚠️ Search event write failed: {e}")
    return {"status": "logged"}

@app.post("/api/search_event/click")
async def log_search_event_click(body: dict, request: Request):
    """その検索でFAQ/動画の結果がクリックされたことを記録（検索IDごと・種別ごとに1回）"""
    try:
        sid = str(body.get("sid") or "")
        kind = str(body.get("type") or "")
        ip = request.client.host if request.client else "unknown"
        if not _SID_RE.match(sid) or kind not in ("faq", "video"):
            return {"status": "ignored"}
        if not _event_rate_ok(ip):
            return {"status": "rate_limited"}
        _append_event({"t": datetime.now(timezone.utc).isoformat(), "sid": sid, "c": kind})
    except Exception as e:
        print(f"⚠️ Search click event write failed: {e}")
    return {"status": "logged"}


# ============================================
# 検索分析（管理画面「ログ閲覧」→「検索分析」）
# ============================================

def _read_search_searches() -> List[Dict[str, Any]]:
    """search_events.jsonl を読み、検索1件=1レコードにまとめる（クリックは検索IDで結合）。"""
    if not SEARCH_EVENTS_PATH.exists():
        return []
    searches: Dict[str, Dict[str, Any]] = {}
    clicks: Dict[str, set] = {}
    bad = 0
    try:
        with open(SEARCH_EVENTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    sid = e["sid"]
                    if "q" in e:
                        if sid in searches:     # 同じ検索IDの重複は最初の1件のみ採用
                            continue
                        dt = datetime.fromisoformat(str(e["t"]).replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        searches[sid] = {
                            "sid": sid, "dt": dt.astimezone(JST), "query": e["q"],
                            "faq": e.get("f"), "video": e.get("v"), "lang": e.get("lang", ""),
                        }
                    elif "c" in e:
                        clicks.setdefault(sid, set()).add(e["c"])
                except Exception:
                    bad += 1
    except Exception as ex:
        print(f"❌ Failed to read search events: {ex}")
        return []
    if bad:
        print(f"⚠️ search_events.jsonl: skipped {bad} bad lines")
    for sid, s in searches.items():
        s["clicks"] = clicks.get(sid, set())
    return list(searches.values())

def _parse_ymd(s: str, name: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(400, f"{name} は YYYY-MM-DD 形式で指定してください")

def _resolve_range(date_from: str, date_to: str):
    today = datetime.now(JST).date()
    d_to = _parse_ymd(date_to, "to") if date_to else today
    d_from = _parse_ymd(date_from, "from") if date_from else d_to - timedelta(days=29)
    if d_from > d_to:
        raise HTTPException(400, "開始日が終了日より後になっています")
    if (d_to - d_from).days > 730:
        raise HTTPException(400, "期間は最大2年（730日）までです")
    return d_from, d_to

def _kpi(searches: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(searches)
    clicked = sum(1 for s in searches if s["clicks"])
    faq_measured = [s for s in searches if s["faq"] is not None]
    zero_faq = sum(1 for s in faq_measured if s["faq"] == 0)
    zero_video = sum(1 for s in searches if s["video"] == 0)
    # 「どちらも0件」= 動画0件 かつ（FAQ0件 または FAQ検索が無効）
    zero_both = sum(1 for s in searches if s["video"] == 0 and (s["faq"] in (0, None)))
    uniq = len({normalize_text(s["query"]) for s in searches})
    def rate(a, b):
        return round(a / b * 100, 1) if b else None
    return {
        "searches": n, "unique_queries": uniq, "clicked": clicked,
        "click_rate": rate(clicked, n),
        "zero_faq": zero_faq, "zero_faq_rate": rate(zero_faq, len(faq_measured)),
        "zero_video": zero_video, "zero_video_rate": rate(zero_video, n),
        "zero_both": zero_both, "zero_both_rate": rate(zero_both, n),
    }

def _in_range(dt: datetime, d_from, d_to) -> bool:
    return d_from <= dt.date() <= d_to

async def build_search_analytics(date_from: str = "", date_to: str = "") -> Dict[str, Any]:
    d_from, d_to = _resolve_range(date_from, date_to)
    span = (d_to - d_from).days + 1
    p_to = d_from - timedelta(days=1)
    p_from = p_to - timedelta(days=span - 1)

    all_searches = _read_search_searches()
    cur = [s for s in all_searches if _in_range(s["dt"], d_from, d_to)]
    prev = [s for s in all_searches if _in_range(s["dt"], p_from, p_to)]
    data_start = min((s["dt"] for s in all_searches), default=None)

    # --- 日別 / 時間帯別 / 曜日別 ---
    daily_map = {d_from + timedelta(days=i): {"searches": 0, "clicked": 0, "zero_both": 0} for i in range(span)}
    hourly = [0] * 24
    weekday = [0] * 7   # 0=月曜
    for s in cur:
        d = daily_map[s["dt"].date()]
        d["searches"] += 1
        if s["clicks"]:
            d["clicked"] += 1
        if s["video"] == 0 and s["faq"] in (0, None):
            d["zero_both"] += 1
        hourly[s["dt"].hour] += 1
        weekday[s["dt"].weekday()] += 1
    daily = [{"day": k.isoformat(), **v} for k, v in sorted(daily_map.items())]

    # --- キーワード単位の集計 ---
    groups: Dict[str, Dict[str, Any]] = {}
    for s in sorted(cur, key=lambda x: x["dt"]):
        key = normalize_text(s["query"])
        g = groups.setdefault(key, {
            "query": s["query"], "count": 0, "clicked": 0, "faq_zero": 0, "video_zero": 0,
            "both_zero": 0, "faq_hits": 0, "faq_n": 0, "video_hits": 0, "last_seen": ""})
        g["query"] = s["query"]                    # 直近の表記を表示
        g["count"] += 1
        g["last_seen"] = s["dt"].strftime("%Y-%m-%d %H:%M")
        g["video_hits"] += s["video"] or 0
        if s["clicks"]:
            g["clicked"] += 1
        if s["faq"] is not None:
            g["faq_n"] += 1
            g["faq_hits"] += s["faq"]
            if s["faq"] == 0:
                g["faq_zero"] += 1
        if s["video"] == 0:
            g["video_zero"] += 1
        if s["video"] == 0 and s["faq"] in (0, None):
            g["both_zero"] += 1

    def brief(g):
        return {
            "query": g["query"], "count": g["count"], "clicked": g["clicked"],
            "click_rate": round(g["clicked"] / g["count"] * 100, 1),
            "faq_zero": g["faq_zero"], "video_zero": g["video_zero"], "both_zero": g["both_zero"],
            "avg_faq_hits": round(g["faq_hits"] / g["faq_n"], 1) if g["faq_n"] else None,
            "avg_video_hits": round(g["video_hits"] / g["count"], 1),
            "last_seen": g["last_seen"],
        }
    rows = [brief(g) for g in groups.values()]
    top_queries = sorted(rows, key=lambda r: (-r["count"], r["query"]))[:50]
    zero_hit = sorted([r for r in rows if r["faq_zero"] or r["video_zero"]],
                      key=lambda r: (-(r["both_zero"] or 0), -r["count"], r["query"]))[:200]
    # クリックなし: ヒットはあったのにクリックされなかった検索が1回以上あるキーワード
    no_click = []
    for g in groups.values():
        no_click_n = g["count"] - g["clicked"] - g["both_zero"]
        if no_click_n > 0:
            r = brief(g)
            r["no_click"] = no_click_n
            no_click.append(r)
    no_click = sorted(no_click, key=lambda r: (-r["no_click"], r["query"]))[:200]

    # --- 期間内のクリックランキング（既存クリックログ） ---
    await state.ensure_faq_loaded()
    await state.ensure_video_loaded()
    faq_counter: Counter = Counter()
    vid_counter: Counter = Counter()
    for r in parse_logs():
        dt = r["dt"]
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if not _in_range(dt.astimezone(JST), d_from, d_to):
            continue
        if r.get("result_type") == "faq" and r.get("result_id"):
            faq_counter[r["result_id"]] += 1
        elif r.get("result_type") == "video" and r.get("result_id"):
            vid_counter[r["result_id"]] += 1
    faq_by_id = {i.get("id"): i for i in state.faq_items_flat}
    vid_by_id = {v.get("video_id"): v for v in state.videos}
    faq_ranking = [{"id": k, "question": faq_by_id[k].get("question", ""), "category": faq_by_id[k].get("category", ""),
                    "click_count": c} for k, c in faq_counter.most_common(30) if k in faq_by_id][:10]
    video_ranking = [{"video_id": k, "title": vid_by_id[k].get("title", ""), "click_count": c}
                     for k, c in vid_counter.most_common(30) if k in vid_by_id][:10]

    return {
        "from": d_from.isoformat(), "to": d_to.isoformat(), "days": span,
        "prev_range": {"from": p_from.isoformat(), "to": p_to.isoformat()},
        "has_data": bool(all_searches),
        "data_start": data_start.strftime("%Y-%m-%d") if data_start else None,
        "kpi": _kpi(cur), "prev": _kpi(prev) if prev else None,
        "daily": daily, "hourly": hourly, "weekday": weekday,
        "top_queries": top_queries, "zero_hit": zero_hit, "no_click": no_click,
        "faq_ranking": faq_ranking, "video_ranking": video_ranking,
    }

@app.get("/admin/api/analytics/overview", dependencies=[Depends(verify_admin)])
async def analytics_overview(date_from: str = Query("", alias="from"), date_to: str = Query("", alias="to")):
    """検索分析の集計（期間指定。省略時は直近30日）"""
    return await build_search_analytics(date_from, date_to)

def xl_safe(v):
    """Excel・CSVで「=」「+」「-」「@」等で始まる文字列が数式として扱われるのを防ぐ。
    お客様の検索語は自由入力なので、悪意ある値（例: =HYPERLINK(...)）が数式として動かないよう、
    先頭に ' を付けて文字列として保存する。数値・日付はそのまま。"""
    if isinstance(v, str) and v and v[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def build_report_xlsx(a: Dict[str, Any]) -> bytes:
    """検索分析をExcelレポート（複数シート）にして返す。"""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="0066CC")
    head_font = Font(bold=True, color="FFFFFF")

    def sheet(title, headers, rows, widths):
        ws = wb.create_sheet(title)
        ws.append(headers)
        for c in ws[1]:
            c.fill, c.font = head_fill, head_font
            c.alignment = Alignment(horizontal="center", vertical="center")
        for r in rows:
            ws.append([xl_safe(v) for v in r])
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A2"
        return ws

    # --- サマリー ---
    ws = wb.active
    ws.title = "サマリー"
    ws["A1"] = "サポート検索 分析レポート"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = f"対象期間: {a['from']} 〜 {a['to']}（{a['days']}日間）"
    ws["A3"] = f"前期間: {a['prev_range']['from']} 〜 {a['prev_range']['to']}"
    ws["A4"] = f"作成日時: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')} (JST)"
    ws.append([])
    ws.append(["指標", "今期間", "前期間", "増減"])
    for c in ws[6]:
        c.fill, c.font = head_fill, head_font
    k, p = a["kpi"], a["prev"] or {}
    items = [("検索回数", "searches", ""), ("ユニークキーワード数", "unique_queries", ""),
             ("結果をクリックした検索", "clicked", ""), ("クリック率(%)", "click_rate", "%"),
             ("FAQ 0件の検索", "zero_faq", ""), ("FAQ 0件率(%)", "zero_faq_rate", "%"),
             ("動画 0件の検索", "zero_video", ""), ("動画 0件率(%)", "zero_video_rate", "%"),
             ("FAQ・動画とも0件の検索", "zero_both", ""), ("両方0件率(%)", "zero_both_rate", "%")]
    for label, key, _ in items:
        cur_v, prev_v = k.get(key), p.get(key)
        diff = (round(cur_v - prev_v, 1) if isinstance(cur_v, (int, float)) and isinstance(prev_v, (int, float)) else "")
        ws.append([label, cur_v if cur_v is not None else "-", prev_v if prev_v is not None else "-", diff])
    ws.column_dimensions["A"].width = 32
    for col in "BCD":
        ws.column_dimensions[col].width = 14
    ws.append([])
    ws.append(["※「ヒットなし」「クリックなし」の分析は、検索実行ログの記録開始日以降のデータが対象です。"])
    if a.get("data_start"):
        ws.append([f"検索実行ログの記録開始日: {a['data_start']}"])

    sheet("日別推移", ["日付", "検索回数", "クリックあり", "両方0件"],
          [[d["day"], d["searches"], d["clicked"], d["zero_both"]] for d in a["daily"]], [14, 12, 14, 12])
    wd = ["月", "火", "水", "木", "金", "土", "日"]
    ws2 = sheet("時間帯・曜日別", ["時間帯(JST)", "検索回数"], [[f"{h:02d}時台", n] for h, n in enumerate(a["hourly"])], [14, 12])
    ws2["D1"], ws2["E1"] = "曜日", "検索回数"
    for c in (ws2["D1"], ws2["E1"]):
        c.fill, c.font = head_fill, head_font
    for i, n in enumerate(a["weekday"]):
        ws2.cell(row=i + 2, column=4, value=wd[i])
        ws2.cell(row=i + 2, column=5, value=n)
    sheet("人気キーワードTOP50", ["キーワード", "検索回数", "クリックあり", "クリック率(%)", "平均FAQ件数", "平均動画件数"],
          [[r["query"], r["count"], r["clicked"], r["click_rate"], r["avg_faq_hits"] if r["avg_faq_hits"] is not None else "-", r["avg_video_hits"]]
           for r in a["top_queries"]], [40, 12, 14, 14, 14, 14])
    sheet("ヒットなしキーワード", ["キーワード", "検索回数", "FAQ0件", "動画0件", "両方0件", "最終検索日時"],
          [[r["query"], r["count"], r["faq_zero"], r["video_zero"], r["both_zero"], r["last_seen"]] for r in a["zero_hit"]],
          [40, 12, 10, 10, 10, 20])
    sheet("クリックなしキーワード", ["キーワード", "クリックなしの検索回数", "検索回数", "平均FAQ件数", "平均動画件数", "最終検索日時"],
          [[r["query"], r["no_click"], r["count"], r["avg_faq_hits"] if r["avg_faq_hits"] is not None else "-", r["avg_video_hits"], r["last_seen"]]
           for r in a["no_click"]], [40, 20, 12, 14, 14, 20])
    sheet("FAQクリックランキング", ["順位", "FAQ ID", "質問", "カテゴリ", "クリック数"],
          [[i + 1, r["id"], r["question"], r["category"], r["click_count"]] for i, r in enumerate(a["faq_ranking"])], [8, 18, 60, 18, 12])
    sheet("動画クリックランキング", ["順位", "動画ID", "タイトル", "クリック数"],
          [[i + 1, r["video_id"], r["title"], r["click_count"]] for i, r in enumerate(a["video_ranking"])], [8, 16, 70, 12])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()

@app.get("/admin/api/analytics/report.xlsx", dependencies=[Depends(verify_admin)])
async def analytics_report_xlsx(date_from: str = Query("", alias="from"), date_to: str = Query("", alias="to")):
    """検索分析のExcelレポートをダウンロード"""
    a = await build_search_analytics(date_from, date_to)
    try:
        data = build_report_xlsx(a)
    except ImportError:
        raise HTTPException(501, "Excel出力ライブラリ(openpyxl)が未導入です。requirements.txt を確認してください")
    fname = f"search_report_{a['from'].replace('-', '')}-{a['to'].replace('-', '')}.xlsx"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ============================================
# 検索精度の計測（管理画面「精度計測」）
# ============================================
# 考え方:
#   「この検索語なら、この結果が正解」という問題集を用意し、今の検索で正解が何位に出るかを数える。
#   問題集は2種類:
#     ① 自動: クリックログ（search_logs.csv）から。「その検索語でお客様がクリックした結果」を正解とみなす
#     ② 手動: 管理者が登録した問題（eval_cases.json）。クリックが少ない重要な質問や、新規FAQの確認用
#   指標:
#     1位正解率 … 正解が1位に出た割合
#     上位3件/10件正解率 … 正解が3位/10位以内に出た割合（お客様が最初に目にする範囲）
#     MRR … 正解の順位の逆数の平均（1位=1.0、2位=0.5、…。順位の良さを1つの数字で表す）
#     見つからない率 … 正解が検索結果に1件も出なかった割合
#   注意: ①はいまの検索で表示された結果からしか選ばれないため、いまの方式に少し有利に出る。
#         方式の比較は同じ問題集・同じ時点で行い、②の手動問題も併用するのがよい。

EVAL_MAX_AUTO_CASES = 300      # 種別ごとの自動問題の上限（クリック数の多い検索語から採用。計測時間を抑えるため）
EVAL_HISTORY_KEEP = 50
_eval_lock = threading.Lock()
_eval_running = {"running": False, "started": None, "label": ""}


def _video_key(v: Dict[str, Any]) -> str:
    vid = str(v.get("video_id") or "").strip()
    if vid:
        return vid
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", str(v.get("url") or "")) or re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", str(v.get("url") or ""))
    return m.group(1) if m else ""


def _read_json_list(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ Failed to read {path.name}: {e}")
        return []


def _write_json_atomic(path: pathlib.Path, data) -> None:
    """書き込み途中で落ちても元ファイルを壊さないよう、一時ファイルに書いてから置き換える"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def build_eval_cases(source: str = "all") -> Dict[str, Any]:
    """計測用の問題集を作る。戻り値: {"cases": [...], "skipped": {...}}"""
    faq_by_id = {str(i.get("id")): i for i in state.faq_items_flat if i.get("id")}
    vid_by_key = {}
    for v in state.videos:
        k = _video_key(v)
        if k:
            vid_by_key[k] = v
    cases: List[Dict[str, Any]] = []
    skipped = {"deleted_target": 0, "short_query": 0}

    if source in ("all", "auto"):
        groups: Dict[tuple, Dict[str, Any]] = {}
        for r in parse_logs():
            t, q, rid = r.get("result_type"), (r.get("query") or "").strip(), (r.get("result_id") or "").strip()
            if t not in ("faq", "video") or not q or not rid:
                continue
            key = (t, normalize_text(q))
            g = groups.setdefault(key, {"type": t, "query": q, "clicks": 0, "ids": Counter()})
            g["clicks"] += 1
            g["ids"][rid] += 1
        for t in ("faq", "video"):
            gs = sorted([g for (tt, _), g in groups.items() if tt == t], key=lambda g: -g["clicks"])
            n = 0
            for g in gs:
                if n >= EVAL_MAX_AUTO_CASES:
                    break
                if len(normalize_text(g["query"])) < 2:
                    skipped["short_query"] += 1
                    continue
                table = faq_by_id if t == "faq" else vid_by_key
                ids = [i for i in g["ids"] if i in table]
                if not ids:
                    skipped["deleted_target"] += 1     # 正解のFAQ/動画が削除済み
                    continue
                lang = table[ids[0]].get("language", "ja") or "ja"
                cases.append({"source": "auto", "type": t, "query": g["query"], "expected": ids,
                              "language": lang, "clicks": g["clicks"]})
                n += 1

    if source in ("all", "manual"):
        for c in _read_json_list(EVAL_CASES_PATH):
            t = c.get("type")
            table = faq_by_id if t == "faq" else vid_by_key
            ids = [str(i) for i in (c.get("expected") or []) if str(i) in table]
            if t not in ("faq", "video") or not c.get("query"):
                continue
            if not ids:
                skipped["deleted_target"] += 1
                continue
            cases.append({"source": "manual", "id": c.get("id"), "type": t, "query": c["query"], "expected": ids,
                          "language": c.get("language") or table[ids[0]].get("language", "ja") or "ja"})
    return {"cases": cases, "skipped": skipped}


def _rss_mb() -> Optional[float]:
    """このプロセスのメモリ使用量(MB)。Linux(/proc)のみ。取れなければNone"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return None


def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round((len(v) - 1) * p))))
    return round(v[i], 3)


def run_eval_sync(cases: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """問題集をすべて検索して指標を計算する（重いので別スレッドで実行する）"""
    per_type = {t: {"n": 0, "hit1": 0, "hit3": 0, "hit10": 0, "rr": 0.0, "miss": 0, "cut_by_threshold": 0,
                    "ms": [], "correct_sem": []} for t in ("faq", "video")}
    details = []
    faq_title = {str(i.get("id")): i.get("question", "") for i in state.faq_items_flat}
    vid_title = {_video_key(v): v.get("title", "") for v in state.videos}
    for c in cases:
        t = c["type"]
        if t == "faq" and not state.faq_index:
            continue
        if t == "video" and not state.video_index:
            continue
        t0 = time.perf_counter()
        try:
            if t == "faq":
                res = faq_search_core(c["query"], c["language"], config, include_rejected=True, verbose=False)
                key = lambda x: str(x.get("id"))
            else:
                res = video_search_core(c["query"], c["language"], config, include_rejected=True, verbose=False)
                key = _video_key
        except Exception as e:
            details.append({**c, "error": str(e)})
            continue
        elapsed = (time.perf_counter() - t0) * 1000
        st = per_type[t]
        st["n"] += 1
        st["ms"].append(elapsed)
        expected = set(c["expected"])
        shown = [x for x in res if x.get("_passed", True)]
        rank = next((i + 1 for i, x in enumerate(shown) if key(x) in expected), None)
        # 正解の意味検索スコア（しきい値の目安づくり用）。落ちた候補も含めて探す
        cand = next((x for x in res if key(x) in expected), None)
        if cand is not None:
            st["correct_sem"].append(float(cand.get("semantic_score", 0)))
        if rank is None:
            st["miss"] += 1
            if cand is not None and cand.get("_passed") is False:
                st["cut_by_threshold"] += 1        # 候補には上がったが、しきい値で表示されなかった
        else:
            st["rr"] += 1.0 / rank
            st["hit1"] += rank <= 1
            st["hit3"] += rank <= 3
            st["hit10"] += rank <= 10
        top = shown[0] if shown else None
        details.append({
            "source": c["source"], "type": t, "query": c["query"], "expected": c["expected"],
            "rank": rank, "total": len(shown), "clicks": c.get("clicks"),
            "top1": (top.get("question") if t == "faq" else top.get("title")) if top else None,
            "expected_title": (faq_title if t == "faq" else vid_title).get(c["expected"][0], ""),
        })

    def pct(a, n):
        return round(a / n * 100, 1) if n else None
    metrics = {}
    for t, st in per_type.items():
        n = st["n"]
        metrics[t] = {
            "cases": n,
            "hit1": pct(st["hit1"], n), "hit3": pct(st["hit3"], n), "hit10": pct(st["hit10"], n),
            "mrr": round(st["rr"] / n, 3) if n else None,
            "miss": pct(st["miss"], n), "cut_by_threshold": st["cut_by_threshold"],
            "avg_ms": round(sum(st["ms"]) / n, 1) if n else None,
            "p95_ms": _percentile(st["ms"], 0.95),
            # しきい値の目安: 正解の意味検索スコアの分布（下位10%点を下回る値にすると、正解の約9割が表示対象に残る）
            "correct_semantic_p10": _percentile(st["correct_sem"], 0.10),
            "correct_semantic_p50": _percentile(st["correct_sem"], 0.50),
        }
    # 失敗した（順位が悪い）問題を上に
    details.sort(key=lambda d: (0 if d.get("rank") is None else 1, -(d.get("rank") or 0)))
    return {"metrics": metrics, "details": details}


def _eval_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": EMBEDDING_MODEL,
        "query_prefix": QUERY_PREFIX, "passage_prefix": PASSAGE_PREFIX,
        "similarity_threshold": config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD),
        "semantic_weight": config.get("semantic_weight", SEMANTIC_WEIGHT),
        "title_weight": config.get("title_weight", TITLE_WEIGHT),
        "ja_tokenizer_enabled": bool(config.get("ja_tokenizer_enabled", False)),
        "ja_tokenizer_available": ja_text.available(),
    }


@app.get("/admin/api/eval/status", dependencies=[Depends(verify_admin)])
async def eval_status():
    """計測画面の上部に出す「現在の検索方式」"""
    config = await get_config()
    return {
        "settings": _eval_settings(config),
        "ja_tokenizer": ja_text.status(),
        "model_loaded": state.model_loaded, "model_load_seconds": state.model_load_seconds,
        "video_loaded": state.video_loaded, "faq_loaded": state.faq_loaded,
        "videos": len(state.videos), "video_chunks": len(state.video_chunk_map), "faqs": len(state.faq_items_flat),
        "memory_mb": _rss_mb(),
        "running": _eval_running,
        "video_rebuilding": state.video_loading, "video_progress": state.video_progress,
    }


@app.get("/admin/api/eval/cases", dependencies=[Depends(verify_admin)])
async def eval_cases_list():
    await state.ensure_faq_loaded()
    await state.ensure_video_loaded()
    built = build_eval_cases("all")
    auto = [c for c in built["cases"] if c["source"] == "auto"]
    return {
        "auto": {"faq": sum(1 for c in auto if c["type"] == "faq"), "video": sum(1 for c in auto if c["type"] == "video"),
                 "sample": auto[:30]},
        "manual": _read_json_list(EVAL_CASES_PATH),
        "skipped": built["skipped"],
    }


@app.post("/admin/api/eval/cases", dependencies=[Depends(verify_admin)])
async def eval_cases_add(body: dict):
    """手動の問題を追加: {"type": "faq"|"video", "query": "...", "expected": ["FAQ-ID または 動画ID", ...], "note": ""}"""
    t = str(body.get("type") or "")
    q = re.sub(r"\s+", " ", str(body.get("query") or "")).strip()[:200]
    exp = body.get("expected") or []
    if isinstance(exp, str):
        exp = [x for x in re.split(r"[,\s、]+", exp) if x]
    exp = [str(x).strip() for x in exp if str(x).strip()][:20]
    if t not in ("faq", "video") or not q or not exp:
        raise HTTPException(400, "種別(faq/video)・検索語・正解ID(1つ以上)は必須です")
    # 存在チェック（打ち間違いの早期発見のため）
    await state.ensure_faq_loaded()
    await state.ensure_video_loaded()
    known = {str(i.get("id")) for i in state.faq_items_flat} if t == "faq" else {_video_key(v) for v in state.videos}
    unknown = [x for x in exp if x not in known]
    if unknown:
        raise HTTPException(400, f"存在しないIDがあります: {', '.join(unknown)}")
    cases = _read_json_list(EVAL_CASES_PATH)
    case = {"id": "c" + secrets.token_hex(5), "type": t, "query": q, "expected": exp,
            "language": str(body.get("language") or "")[:5], "note": str(body.get("note") or "")[:200],
            "created_at": datetime.now(timezone.utc).isoformat()}
    cases.append(case)
    _write_json_atomic(EVAL_CASES_PATH, cases)
    return {"status": "ok", "case": case}


@app.delete("/admin/api/eval/cases/{case_id}", dependencies=[Depends(verify_admin)])
async def eval_cases_delete(case_id: str):
    cases = _read_json_list(EVAL_CASES_PATH)
    new = [c for c in cases if c.get("id") != case_id]
    if len(new) == len(cases):
        raise HTTPException(404, "該当する問題がありません")
    _write_json_atomic(EVAL_CASES_PATH, new)
    return {"status": "ok"}


@app.post("/admin/api/eval/run", dependencies=[Depends(verify_admin)])
async def eval_run(body: dict):
    """計測を実行。
    body: {"label": "メモ", "source": "all|auto|manual",
           "overrides": {"ja_tokenizer_enabled": true, "similarity_threshold": 0.3, ...}  ← 本番設定を変えずに試す
           "save": true}"""
    if not _eval_lock.acquire(blocking=False):
        raise HTTPException(409, "別の計測を実行中です。終わってから再度お試しください")
    try:
        _eval_running.update({"running": True, "started": datetime.now(timezone.utc).isoformat(),
                              "label": str(body.get("label") or "")[:100]})
        await state.ensure_faq_loaded()
        await state.ensure_video_loaded()
        config = dict(await get_config())
        allowed = {"ja_tokenizer_enabled": bool, "similarity_threshold": float, "semantic_weight": float, "title_weight": float}
        overrides = {}
        for k, conv in allowed.items():
            if k in (body.get("overrides") or {}):
                try:
                    overrides[k] = conv(body["overrides"][k])
                except Exception:
                    raise HTTPException(400, f"{k} の値が不正です")
        config.update(overrides)
        if config.get("ja_tokenizer_enabled") and not ja_text.available():
            raise HTTPException(400, f"日本語解析ライブラリが使えません（{ja_text.status().get('error')}）。requirements.txt の fugashi / unidic-lite を確認してください")
        source = body.get("source") if body.get("source") in ("all", "auto", "manual") else "all"
        built = build_eval_cases(source)
        if not built["cases"]:
            raise HTTPException(400, "計測できる問題がありません（クリックログがまだ少ないか、手動の問題が未登録です）")
        t0 = time.time()
        result = await asyncio.to_thread(run_eval_sync, built["cases"], config)
        record = {
            "id": "e" + datetime.now(JST).strftime("%Y%m%d%H%M%S") + secrets.token_hex(2),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "label": _eval_running["label"],
            "source": source, "overrides": overrides,
            "settings": _eval_settings(config),
            "metrics": result["metrics"],
            "seconds": round(time.time() - t0, 1),
            "memory_mb": _rss_mb(),
            "skipped": built["skipped"],
            "details": result["details"][:300],
        }
        if body.get("save", True):
            hist = _read_json_list(EVAL_HISTORY_PATH)
            hist.append(record)
            try:
                _write_json_atomic(EVAL_HISTORY_PATH, hist[-EVAL_HISTORY_KEEP:])
            except Exception as e:
                print(f"⚠️ Failed to save eval history: {e}")
                record["save_error"] = str(e)
        return record
    finally:
        _eval_running.update({"running": False, "started": None, "label": ""})
        _eval_lock.release()


@app.get("/admin/api/eval/history", dependencies=[Depends(verify_admin)])
async def eval_history():
    hist = _read_json_list(EVAL_HISTORY_PATH)
    return {"history": [{k: v for k, v in h.items() if k != "details"} for h in reversed(hist)]}


@app.get("/admin/api/eval/history/{eval_id}", dependencies=[Depends(verify_admin)])
async def eval_history_detail(eval_id: str):
    for h in _read_json_list(EVAL_HISTORY_PATH):
        if h.get("id") == eval_id:
            return h
    raise HTTPException(404, "計測結果が見つかりません")


@app.delete("/admin/api/eval/history/{eval_id}", dependencies=[Depends(verify_admin)])
async def eval_history_delete(eval_id: str):
    hist = _read_json_list(EVAL_HISTORY_PATH)
    new = [h for h in hist if h.get("id") != eval_id]
    if len(new) == len(hist):
        raise HTTPException(404, "計測結果が見つかりません")
    _write_json_atomic(EVAL_HISTORY_PATH, new)
    return {"status": "ok"}


@app.get("/admin/eval", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_eval():
    f = admin_path / "eval.html"
    if not f.exists():
        return HTMLResponse("<h1>admin eval.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")


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
async def get_public_synonyms():
    """Synonyms.jsonã‚’è¿”ã™"""
    try:
        # 永続ディスク上の辞書を読む（以前は作業フォルダの synonyms.json を見ており、本番では見つからなかった）
        with open(SYNONYMS_PATH, 'r', encoding='utf-8') as f:
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
