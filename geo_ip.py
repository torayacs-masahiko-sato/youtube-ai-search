# -*- coding: utf-8 -*-
"""
geo_ip.py — IPアドレスから「国」「都道府県」を調べる（検索分析の「利用地域」用）

■ 使うデータ
  DB-IP「IP to City Lite」（無料・月1回更新・CC BY 4.0）
  https://db-ip.com/db/download/ip-to-city-lite
  ライセンス条件として、結果を表示する画面に「IP Geolocation by DB-IP」のリンクが必要（ログ閲覧画面に表示済み）。

■ 動き
  - 起動時に別スレッドで DATA_DIR/geoip/dbip-city-lite.mmdb を確認し、無い・35日以上古い場合はダウンロードする
    （約60MBのダウンロード、展開後約125MB。Persistent Diskに保存するので再起動しても残る）
  - その後は1日1回、更新が必要か確認する
  - データが用意できるまで／ダウンロードに失敗した時は「不明」として扱う。検索そのものには影響しない

■ 個人情報への配慮
  IPアドレスは調べるためにその場で使うだけで、保存しない。記録するのは国コード・国名・都道府県名のみ。

■ 環境変数（どれも任意）
  GEOIP_ENABLED=0      … この機能を止める（地域を記録しない）
  GEOIP_DB_PATH=...    … 別の .mmdb（例: MaxMind GeoLite2-City.mmdb）を使う。指定した場合は自動ダウンロードしない

■ 必要ライブラリ
  maxminddb（requirements.txt に記載）
"""
import gzip
import ipaddress
import os
import pathlib
import shutil
import threading
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

try:
    import maxminddb
except Exception:          # 未インストールでも本体は起動できるようにする
    maxminddb = None

DB_URL_TMPL = "https://download.db-ip.com/free/dbip-city-lite-{ym}.mmdb.gz"
MAX_AGE_DAYS = 35          # 月1回更新なので、35日を超えたら取り直す
CHECK_INTERVAL_SEC = 24 * 3600

_lock = threading.Lock()
_reader = None
_status: Dict[str, Any] = {"state": "未開始", "db_date": None, "error": None, "source": None}
_ip_source_seen: Dict[str, int] = {}   # どのヘッダーからIPを取れたか（Render上での確認用。IPは保存しない）

# ---- 都道府県（DB-IPは英語名のみなので日本語名に変換する） ----
_PREFS = [
    ("hokkaido", "北海道"), ("aomori", "青森県"), ("iwate", "岩手県"), ("miyagi", "宮城県"), ("akita", "秋田県"),
    ("yamagata", "山形県"), ("fukushima", "福島県"), ("ibaraki", "茨城県"), ("tochigi", "栃木県"), ("gunma", "群馬県"),
    ("saitama", "埼玉県"), ("chiba", "千葉県"), ("tokyo", "東京都"), ("kanagawa", "神奈川県"), ("niigata", "新潟県"),
    ("toyama", "富山県"), ("ishikawa", "石川県"), ("fukui", "福井県"), ("yamanashi", "山梨県"), ("nagano", "長野県"),
    ("gifu", "岐阜県"), ("shizuoka", "静岡県"), ("aichi", "愛知県"), ("mie", "三重県"), ("shiga", "滋賀県"),
    ("kyoto", "京都府"), ("osaka", "大阪府"), ("hyogo", "兵庫県"), ("nara", "奈良県"), ("wakayama", "和歌山県"),
    ("tottori", "鳥取県"), ("shimane", "島根県"), ("okayama", "岡山県"), ("hiroshima", "広島県"), ("yamaguchi", "山口県"),
    ("tokushima", "徳島県"), ("kagawa", "香川県"), ("ehime", "愛媛県"), ("kochi", "高知県"), ("fukuoka", "福岡県"),
    ("saga", "佐賀県"), ("nagasaki", "長崎県"), ("kumamoto", "熊本県"), ("oita", "大分県"), ("miyazaki", "宮崎県"),
    ("kagoshima", "鹿児島県"), ("okinawa", "沖縄県"),
]
PREF_ORDER = [ja for _, ja in _PREFS]          # 表示順（北から）
_PREF_BY_EN = dict(_PREFS)
_PREF_BY_EN.update({"gumma": "群馬県", "hyougo": "兵庫県", "kouchi": "高知県", "ooita": "大分県"})


def pref_ja(name_en: str) -> str:
    """'Tokyo' / 'Hyōgo' / 'Osaka-fu' / 'Kyoto Prefecture' → 日本語の都道府県名。分からなければ ''"""
    if not name_en:
        return ""
    s = unicodedata.normalize("NFKD", name_en).encode("ascii", "ignore").decode().lower().strip()
    for suf in (" prefecture", "-ken", " ken", "-fu", " fu", "-to", " to", "-do"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    s = s.replace(" ", "").replace("-", "")
    return _PREF_BY_EN.get(s, "")


# ---- 実際のお客様のIPを取り出す ----
_IP_HEADERS = ("true-client-ip", "cf-connecting-ip", "x-forwarded-for", "x-real-ip")


def _public_ip(value: str) -> Optional[str]:
    try:
        ip = ipaddress.ip_address(value.strip().split("%")[0])
    except Exception:
        return None
    if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return None
    return str(ip)


def client_ip(request) -> Optional[str]:
    """Render等のプロキシ経由では request.client.host がプロキシのアドレスになるため、
    転送ヘッダーから「最初の公開IP」を取り出す。どこから取れたかは件数だけ記録する（確認用）。
    ※ ヘッダーは偽装できるが、用途は統計なので許容する（認証・制限には使わない）。"""
    for h in _IP_HEADERS:
        v = request.headers.get(h)
        if not v:
            continue
        for part in v.split(","):
            ip = _public_ip(part)
            if ip:
                _ip_source_seen[h] = _ip_source_seen.get(h, 0) + 1
                return ip
    host = request.client.host if request.client else ""
    ip = _public_ip(host) if host else None
    _ip_source_seen["direct" if ip else "none"] = _ip_source_seen.get("direct" if ip else "none", 0) + 1
    return ip


# ---- 調べる ----
def lookup(ip: Optional[str]) -> Dict[str, str]:
    """{'cc': 国コード, 'cn': 国名(日本語), 'pref': 都道府県(日本のみ)}。分からない項目は ''"""
    out = {"cc": "", "cn": "", "pref": ""}
    if not ip or _reader is None:
        return out
    try:
        with _lock:
            rec = _reader.get(ip) if _reader is not None else None
    except Exception:
        return out
    if not rec:
        return out
    c = rec.get("country") or {}
    names = c.get("names") or {}
    out["cc"] = str(c.get("iso_code") or "")[:2]
    out["cn"] = str(names.get("ja") or names.get("en") or out["cc"])[:40]
    if out["cc"] == "JP":
        subs = rec.get("subdivisions") or []
        if subs:
            sn = subs[0].get("names") or {}
            out["pref"] = pref_ja(sn.get("en") or "")
    return out


def status() -> Dict[str, Any]:
    s = dict(_status)
    s["ip_source"] = dict(_ip_source_seen)
    s["attribution"] = "IP Geolocation by DB-IP (https://db-ip.com)"
    return s


# ---- データの用意と更新 ----
def _open(path: pathlib.Path) -> None:
    global _reader
    # MODE_FILE: ファイルを都度読む方式。メモリをほぼ使わない（検索1回につき1回しか引かないので速度は十分）
    new = maxminddb.open_database(str(path), maxminddb.MODE_FILE)
    with _lock:
        old, _reader = _reader, new
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
    meta = new.metadata()
    _status["db_date"] = datetime.fromtimestamp(meta.build_epoch, timezone.utc).strftime("%Y-%m-%d")
    _status["state"] = "利用可能"


def _download(dest: pathlib.Path) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    last_err = None
    # 月初はその月のファイルがまだ無いことがあるので、前月分も試す
    for d in (now, now.replace(day=1) - timedelta(days=1)):
        url = DB_URL_TMPL.format(ym=d.strftime("%Y-%m"))
        gz_tmp = dest.with_suffix(".mmdb.gz.tmp")
        tmp = dest.with_suffix(".mmdb.tmp")
        try:
            _status["state"] = "ダウンロード中"
            with requests.get(url, stream=True, timeout=(15, 300)) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {url}")
                with open(gz_tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            with gzip.open(gz_tmp, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            maxminddb.open_database(str(tmp), maxminddb.MODE_FILE).close()   # 壊れていないか確認
            os.replace(tmp, dest)       # 途中で失敗しても、使用中のファイルは壊さない
            _status["error"] = None
            print(f"🌏 GeoIP database downloaded: {url}")
            return
        except Exception as e:
            last_err = e
            print(f"⚠️ GeoIP download failed: {e}")
        finally:
            for p in (gz_tmp, tmp):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
    raise RuntimeError(str(last_err))


def _ensure(path: pathlib.Path, auto_download: bool) -> None:
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < MAX_AGE_DAYS * 86400
    if auto_download and not fresh:
        try:
            _download(path)
        except Exception as e:
            _status["error"] = f"ダウンロード失敗: {e}"
            if not path.exists():
                _status["state"] = "利用不可（データなし）"
                return
            # 古いデータが残っていればそれで続ける（翌日また取得を試みる）
    if path.exists() and (_reader is None or not fresh):
        _open(path)
    elif not path.exists():
        _status["state"] = "利用不可（データなし）"


def start(data_dir: pathlib.Path) -> None:
    """起動時に1回呼ぶ。準備は別スレッドで行い、起動を待たせない。"""
    if os.getenv("GEOIP_ENABLED", "1") == "0":
        _status["state"] = "停止中（GEOIP_ENABLED=0）"
        return
    if maxminddb is None:
        _status["state"] = "利用不可（maxminddb 未インストール）"
        return
    custom = os.getenv("GEOIP_DB_PATH", "").strip()
    path = pathlib.Path(custom) if custom else pathlib.Path(data_dir) / "geoip" / "dbip-city-lite.mmdb"
    _status["source"] = "custom" if custom else "DB-IP Lite"

    def loop():
        while True:
            try:
                _ensure(path, auto_download=not custom)
            except Exception as e:
                _status["error"] = str(e)
                print(f"⚠️ GeoIP setup error: {e}")
            time.sleep(CHECK_INTERVAL_SEC)

    threading.Thread(target=loop, name="geoip", daemon=True).start()
