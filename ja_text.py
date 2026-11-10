"""
日本語解析（形態素解析）ユーティリティ

目的:
  検索キーワードと、動画タイトル・FAQ本文との「キーワード一致度」を、
  文字2文字単位（bi-gram）ではなく「単語単位」で判定できるようにする。

  例) 「プリンタ 印刷できない」
      bi-gram: 「プリ」「リン」「ンタ」「印刷」「刷で」… → 関係のない文字の並びも一致扱いになりやすい
      単語単位: 「プリンター」「印刷」 → 意味のある語だけで判定。
               「プリンタ」と「プリンター」、「サーバ」と「サーバー」も同じ語として扱う（辞書の見出し語で統一）

使用ライブラリ:
  fugashi（MeCabのPython版）＋ unidic-lite（辞書）。requirements.txt に記載。
  インストールされていない・読み込みに失敗した場合は available() が False を返し、
  呼び出し側は従来のbi-gram方式のまま動く（検索が止まらないことを優先）。
"""
from functools import lru_cache
from typing import List, Optional, Set
import re
import threading
import unicodedata

_tagger = None
_tagger_error: Optional[str] = None
_tagger_lock = threading.Lock()   # MeCabのTaggerはスレッド間で同時に使わない

# 検索の手がかりにならない語（見出し語）。
# 「〜する」「〜できる」「〜方法」などはどの文書にも出てくるため、一致しても順位の参考にならない。
STOP_LEMMAS: Set[str] = {
    "為る", "する", "有る", "ある", "居る", "いる", "成る", "なる", "出来る", "できる",
    "事", "こと", "物", "もの", "様", "よう", "為", "ため", "方", "方法", "仕方", "遣り方", "やり方",
    "時", "とき", "所", "ところ", "場合", "下さる", "くださる", "頂く", "いただく", "貰う", "もらう",
    "欲しい", "ほしい", "良い", "よい", "いい", "思う", "分かる", "わかる", "教える", "知る",
    "どう", "何", "なに", "なぜ", "どうして", "どこ", "いつ", "これ", "それ", "あれ",
    "the", "a", "an", "to", "of", "in", "on", "for", "is", "are", "how", "what", "can", "do", "i",
}

# 採用する品詞（名詞・動詞・形容詞など内容を表す語）。助詞・助動詞・記号は除外。
CONTENT_POS = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "代名詞", "接頭辞"}
_SKIP_POS = {"助詞", "助動詞", "補助記号", "記号", "空白", "感動詞", "接続詞", "連体詞"}


def _get_tagger():
    global _tagger, _tagger_error
    if _tagger is not None or _tagger_error is not None:
        return _tagger
    with _tagger_lock:
        if _tagger is None and _tagger_error is None:
            try:
                import fugashi  # noqa: WPS433（遅延import: 未インストールでも本体は起動させるため）
                _tagger = fugashi.Tagger()
                print("✅ Japanese tokenizer (fugashi + unidic-lite) loaded")
            except Exception as e:  # ImportError / 辞書が見つからない 等
                _tagger_error = f"{type(e).__name__}: {e}"
                print(f"⚠️ Japanese tokenizer unavailable, falling back to bi-gram: {_tagger_error}")
    return _tagger


def available() -> bool:
    """形態素解析が使えるか（初回呼び出し時に読み込む）"""
    return _get_tagger() is not None


def status() -> dict:
    available()
    return {"available": _tagger is not None, "error": _tagger_error}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def _lemma(word) -> str:
    """辞書の見出し語。UniDicは外来語に「プリンター-printer」の形で英語を付けるので前半だけ使う"""
    try:
        lemma = word.feature.lemma
    except Exception:
        lemma = None
    if not lemma or lemma == "*":
        return word.surface.lower()
    return lemma.split("-")[0].lower() or word.surface.lower()


@lru_cache(maxsize=20000)
def tokens(text: str) -> tuple:
    """内容語の (見出し語, 表層形) のタプル列を返す。重複は除く。解析できない時は空。"""
    tagger = _get_tagger()
    t = _norm(text)
    if tagger is None or not t:
        return tuple()
    out = []
    seen = set()
    with _tagger_lock:
        words = [(w.surface, _lemma(w), getattr(w.feature, "pos1", "") or "") for w in tagger(t)]
    for surface, lemma, pos1 in words:
        if pos1 in _SKIP_POS or (pos1 and pos1 not in CONTENT_POS):
            continue
        if not surface.strip() or lemma in STOP_LEMMAS or surface in STOP_LEMMAS:
            continue
        if len(surface) == 1 and not re.match(r"[一-龥a-z0-9]", surface):
            continue   # 1文字のひらがな・カタカナは手がかりにならない
        if lemma in seen:
            continue
        seen.add(lemma)
        out.append((lemma, surface))
    return tuple(out)


@lru_cache(maxsize=20000)
def lemma_set(text: str) -> frozenset:
    return frozenset(l for l, _ in tokens(text))


def token_coverage(query: str, doc: str) -> Optional[float]:
    """検索語の内容語のうち、文書に含まれる割合（0.0〜1.0）。
    内容語が1つも取れない（例: 「どうすればいい？」）・解析不可の時は None（呼び出し側で従来方式）。
    一致の判定は ①見出し語が一致（表記ゆれ吸収） または ②表層形が文書に文字列として含まれる（複合語対策）。"""
    q = tokens(query)
    if not q:
        return None
    doc_lemmas = lemma_set(doc)
    doc_norm = _norm(doc)
    hit = sum(1 for lemma, surface in q if lemma in doc_lemmas or surface in doc_norm)
    return hit / len(q)
