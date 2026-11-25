# -*- coding: utf-8 -*-
"""
video_keywords.py — 動画のキーワード自動生成

■ 何をするか
  動画のタイトル・説明文・文字起こしから、その動画を特徴づける語を最大10個えらぶ。
  例）「パターン作成」「縫い代」「グレーディングピッチ」「PDF出力」

■ えらび方（外部サービスは使わず、サーバー内で完結）
  1. 日本語解析（fugashi）で名詞を取り出す。続けて並ぶ名詞は1語にまとめる
     （「パターン」「作成」「画面」→「パターン作成画面」）。
     前後の「方法」「場合」などの意味の薄い語は外す（「設定方法」→「設定」）
  2. 点数 = その動画の中での出現回数 × 他の動画にはあまり出てこない度合い（TF-IDF）
     - タイトルは3倍、説明文は2倍、文字起こしは1倍として数える（タイトルが一番内容を表すため）
     - ほとんどの動画に出てくる語（「クリック」「画面」等）は点数が下がるので自然に外れる
  3. 点数の高い順にえらぶ。片方がもう片方を含む語（「パターン」と「パターン作成」）は、
     点数の高い方だけ残す（似た語ばかり並ばないように）

■ 日本語解析が使えない環境では
  カタカナ・漢字・英数字のまとまりを取り出す簡易方式で動く（精度は落ちるが止まらない）。

■ 英語の動画（language=en）
  英単語と2語の組み合わせから、同じ方式でえらぶ。
"""
import hashlib
import math
import re
import threading
import unicodedata
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ja_text

MAX_KEYWORDS = 10
FIELD_WEIGHTS = (("title", 3.0), ("description", 2.0), ("transcript", 1.0))

# 語のまとまりの前後から外す、意味の薄い語（「設定方法」→「設定」、「サイズごと」→「サイズ」）
EDGE_TRIM = {
    "方法", "仕方", "やり方", "場合", "こと", "事", "もの", "物", "ため", "為", "よう", "様", "時", "とき", "際",
    "部分", "辺", "感じ", "等", "など", "以上", "以下", "今回", "次", "前", "後", "中", "上", "下",
    "ほう", "方", "形", "風", "通り", "とおり", "最初", "最後", "続き", "全部", "全て", "すべて", "今", "先", "あと",
    "まま", "ところ", "所", "はず", "わけ", "たち", "達", "ごと", "毎", "別", "的", "化", "性", "目", "回", "個",
    "番", "点", "分", "秒", "つ", "ご", "お", "手順", "皆さん", "皆様", "みなさん", "本日", "今日", "ここ", "こちら",
}
# 単独ではキーワードにしない語（操作説明の動画ならどれにも出てくる語や、挨拶・チャンネル名など）。
# 複合語の一部としては残る（例:「ピッチ設定」「出力プレビュー」）
EXCLUDE = EDGE_TRIM | {
    "動画", "説明", "紹介", "そちら", "あちら", "これ", "それ", "あれ", "どれ", "何", "左", "右", "右上", "左上",
    "右下", "左下", "一つ", "二つ", "1つ", "2つ", "自分", "私", "我々",
    "操作", "手順", "基本", "画面", "表示", "選択", "指定", "クリック", "ダブルクリック", "メニュー", "ボタン", "実行",
    "入力", "追加", "位置", "付け", "初期", "処理", "対処", "解決", "移動", "方向", "種類", "一覧", "箇所", "外側",
    "内側", "使用", "利用", "機能", "必要", "可能", "状態", "変更", "設定", "確認", "作業", "完了", "開始", "終了",
    "ok", "はい", "えー", "あの", "その", "まず", "では", "それでは", "ご覧", "お願い", "ご紹介", "ご説明", "ご確認",
    "ご視聴", "ご利用", "お客様", "センチ", "ミリ", "メートル", "パーセント", "cm", "mm", "ii", "iii", "iv", "チャンネル", "登録", "視聴", "東レ", "東レacs", "acs", "サポートチャンネル",
}
EN_STOP = set(ja_text.STOP_LEMMAS) | {
    "and", "or", "the", "this", "that", "these", "those", "with", "from", "by", "as", "at", "be", "it", "its",
    "you", "your", "we", "our", "will", "then", "here", "there", "now", "so", "if", "into", "out", "up", "down",
    "also", "just", "click", "select", "video", "channel", "subscribe", "please", "thank", "thanks", "next",
    "first", "let", "lets", "let's", "use", "using", "used", "make", "see", "shown", "show", "one", "two",
    "was", "were", "has", "have", "had", "not", "all", "any", "more", "some", "which", "when", "where", "who",
    "about", "after", "before", "following", "screen", "button", "how", "toray", "acs", "support",
    "add", "enter", "choose", "apply", "open", "shows", "set", "go", "get", "want", "need", "done",
}

_RE_KANA_ONLY = re.compile(r"^[ぁ-ゖー]+$")
_RE_NUM_ONLY = re.compile(r"^[0-9.,:/\-x ]+$")
_RE_ASCII = re.compile(r"^[A-Za-z0-9+#._\-]+$")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def _key(s: str) -> str:
    """同じ語かどうかの判定用（大文字小文字・空白の違いを無視）"""
    return re.sub(r"\s+", "", _norm(s).lower())


def _ok_word(w: str) -> bool:
    k = _key(w)
    if not k or k in EXCLUDE or _RE_NUM_ONLY.match(k) or _RE_KANA_ONLY.match(k):
        return False
    if re.match(r"^[0-9]", k) and not _RE_ASCII.match(k):
        return False      # 「1センチ」「100パーセント」のような数量は使わない（「3D」は残す）
    if len(k) < 2:
        return False
    return True


# ---------------- 語の取り出し ----------------
def _ja_terms(text: str) -> List[str]:
    """日本語の文から、名詞のまとまり（複合語）と単独の名詞を取り出す。"""
    tagger = ja_text._get_tagger()
    text = _norm(text)
    if not text:
        return []
    if tagger is None:
        return _fallback_terms(text)
    with ja_text._tagger_lock:   # MeCabはスレッド間で同時に使わない
        words = [(w.surface, getattr(w.feature, "pos1", "") or "", getattr(w.feature, "pos2", "") or "",
                  bool(getattr(w, "white_space", ""))) for w in tagger(text)]
    out: List[str] = []
    run: List[Tuple[str, bool]] = []   # (表層形, 直前に空白があったか)

    def flush():
        # 前後の意味の薄い語を外してから登録する
        parts = list(run)
        while parts and _key(parts[0][0]) in EDGE_TRIM:
            parts.pop(0)
        while parts and _key(parts[-1][0]) in EDGE_TRIM:
            parts.pop()
        if parts:
            # 英数字どうしの間の空白は残す（「CREACOMPO II」が「CREACOMPOII」にならないように）
            s = parts[0][0]
            for (sf, sp), (prev, _) in zip(parts[1:], parts[:-1]):
                s += (" " if sp and _RE_ASCII.match(sf) and _RE_ASCII.match(prev) else "") + sf
            all_generic = all(_key(sf) in EXCLUDE for sf, _ in parts)   # 「画面右上」「基本操作」など
            if 1 < len(parts) <= 4 and len(_key(s)) <= 20 and not all_generic and _ok_word(s):
                out.append(s)                         # 複合語
            for sf, _ in parts:
                if _ok_word(sf) and not (_RE_ASCII.match(sf) and len(sf) < 2):
                    out.append(sf)                    # 単独の名詞
        run.clear()

    for i, (sf, p1, p2, sp) in enumerate(words):
        is_noun = p1 == "名詞" and p2 in ("普通名詞", "固有名詞", "数詞")
        is_prefix = p1 == "接頭辞" and not run
        is_suffix = p1 == "接尾辞" and p2 == "名詞的" and run
        # 空白をはさんだ日本語どうしは別の語（文字起こしの区切り）として扱う
        if run and sp and not (_RE_ASCII.match(sf) and _RE_ASCII.match(run[-1][0])):
            flush()
        if is_noun or is_suffix or is_prefix:
            if p2 == "数詞" and not run and not (i + 1 < len(words) and words[i + 1][1] == "名詞"):
                continue                              # 単独の数字は使わない
            run.append((sf, sp))
        else:
            flush()
    flush()
    return out


def _fallback_terms(text: str) -> List[str]:
    """日本語解析が使えない時の簡易方式：カタカナ・漢字・英数字のまとまり"""
    pats = (r"[ァ-ヴー]{3,}", r"[一-龥々]{2,8}", r"[A-Za-z][A-Za-z0-9+#.\-]{1,}")
    out = []
    for p in pats:
        out += [m for m in re.findall(p, text) if _ok_word(m)]
    return out


def _en_terms(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#\-]*", _norm(text))
    toks = [w for w in words]
    out = []
    prev_ok = None
    for w in toks:
        ok = w.lower() not in EN_STOP and len(w) >= 3
        if ok:
            out.append(w)
            if prev_ok:
                out.append(prev_ok + " " + w)          # 2語の組み合わせ（例: pattern grading）
        prev_ok = w if ok else None
    return out


def _terms(text: str, language: str) -> List[str]:
    return _en_terms(text) if language == "en" else _ja_terms(text)


# ---------------- 点数づけ ----------------
_COUNT_CACHE: "OrderedDict[str, Tuple[Counter, Dict[str, str], set]]" = OrderedDict()
_COUNT_CACHE_MAX = 5000
_cache_lock = threading.Lock()


def _doc_counts(video: Dict) -> Tuple[Counter, Dict[str, str], set]:
    """動画1件の語ごとの重み付き出現回数・表示用の表記・タイトルに出る語。
    文字起こしの解析は時間がかかるので、内容が同じ動画は前回の結果を使い回す（2回目以降が速い）。"""
    lang = (video.get("language") or "ja").lower()
    h = hashlib.sha1("\x1f".join([lang, str(ja_text.available())] + [str(video.get(f) or "") for f, _ in FIELD_WEIGHTS])
                     .encode("utf-8")).hexdigest()
    with _cache_lock:
        hit = _COUNT_CACHE.get(h)
        if hit is not None:
            _COUNT_CACHE.move_to_end(h)
            return hit
    res = _doc_counts_uncached(video, lang)
    with _cache_lock:
        _COUNT_CACHE[h] = res
        while len(_COUNT_CACHE) > _COUNT_CACHE_MAX:
            _COUNT_CACHE.popitem(last=False)
    return res


def _doc_counts_uncached(video: Dict, lang: str) -> Tuple[Counter, Dict[str, str], set]:
    counts: Counter = Counter()
    display: Dict[str, str] = {}
    in_title = set()
    for field, w in FIELD_WEIGHTS:
        for t in _terms(str(video.get(field) or ""), lang):
            k = _key(t)
            counts[k] += w
            display.setdefault(k, t)       # 最初に出てきた表記（タイトル優先）で表示
            if field == "title":
                in_title.add(k)
    return counts, display, in_title


def build_df(videos: Iterable[Dict]) -> Tuple[Counter, int]:
    """全動画での「その語が出てくる動画の数」（他の動画にどれだけ出るかの判定用）"""
    df: Counter = Counter()
    n = 0
    for v in videos:
        counts, _, _ = _doc_counts(v)
        df.update(counts.keys())
        n += 1
    return df, n


def generate(video: Dict, df: Counter, n_docs: int, max_keywords: int = MAX_KEYWORDS) -> List[str]:
    """動画1件のキーワードを返す（点数の高い順）"""
    counts, display, in_title = _doc_counts(video)
    if not counts:
        return []
    scored = []
    for k, tf in counts.items():
        d = df.get(k, 0)
        if n_docs >= 10 and d / n_docs > 0.5:
            continue                       # 半分以上の動画に出てくる語は特徴にならない
        if (video.get("language") or "ja").lower() == "en" and " " in display[k] and tf < 2 and k not in in_title:
            continue                       # 英語の2語の組み合わせは、2回以上出たものだけ
        idf = math.log((n_docs + 1) / (d + 1)) + 1.0
        s = (1.0 + math.log(tf)) * idf
        if " " in display[k] or len(k) >= 4 and not _RE_ASCII.match(k):
            s *= 1.15                      # 長めの語・複合語は内容をよく表す
        if k in in_title:
            s *= 1.5
        scored.append((s, k))
    scored.sort(key=lambda x: (-x[0], x[1]))
    # 含む・含まれる関係の語は1つにまとめる（似た語ばかり並ばないように）。
    # 長い語（「ライセンスキー」）の点数が短い語（「ライセンス」）の6割以上なら長い方に置き換える（内容がより具体的なため）
    score_of = {k: sc for sc, k in scored}
    picked: List[str] = []
    for sc, k in scored:
        if len(picked) >= max_keywords:
            break
        rel = [p for p in picked if k in p or p in k]
        if not rel:
            picked.append(k)
            continue
        if all(p in k for p in rel) and all(sc >= 0.6 * score_of[p] for p in rel):
            first = picked.index(rel[0])
            picked = [p for p in picked if p not in rel]
            picked.insert(min(first, len(picked)), k)
    return [display[k] for k in picked[:max_keywords]]


def clean_keywords(values: Sequence) -> List[str]:
    """手入力・ファイルのキーワードを整える（空・重複を除く）"""
    out, seen = [], set()
    if isinstance(values, str):
        values = re.split(r"[,、，]", values)
    for v in values or []:
        if v is None:
            continue
        s = _norm(str(v))
        k = _key(s)
        if s and k not in seen:
            seen.add(k)
            out.append(s[:50])
    return out
