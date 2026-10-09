"""
管理画面の認証ユーティリティ（多要素認証対応）

役割:
  - パスワードのハッシュ化（PBKDF2-SHA256）と照合
  - TOTP（Google Authenticator / Microsoft Authenticator 互換）の生成・検証
  - リカバリーコード（スマホ紛失時の予備コード）の発行・照合
  - ログイン後に使う署名付きトークン（セッション）の発行・検証
  - ログイン失敗回数の制限（ロックアウト）
  - users.json の読み書き

なぜ標準ライブラリだけで実装したか:
  追加ライブラリを増やすと、デプロイ失敗（依存関係エラー）のリスクが増えるため。
  TOTPは RFC 6238、パスワードハッシュは PBKDF2 という標準規格で、
  標準ライブラリ(hmac / hashlib)だけで安全に実装できる。

前提:
  - 環境変数 DATA_DIR（Persistent Diskのマウント先）配下に users.json を置く
  - 環境変数 SESSION_SECRET があればそれを署名鍵に使う。
    無ければ DATA_DIR/session_secret.key を自動生成して使う。
"""

import base64
import hashlib
import hmac
import json
import os
import pathlib
import secrets
import struct
import threading
import time
from typing import Optional, Tuple, List
from urllib.parse import quote

# ------------------------------------------------------------
# 設定値
# ------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent
_DATA_DIR_ENV = os.getenv("DATA_DIR", "")
DATA_DIR = pathlib.Path(_DATA_DIR_ENV) if _DATA_DIR_ENV else BASE_DIR
USERS_PATH = DATA_DIR / "users.json"
SESSION_KEY_PATH = DATA_DIR / "session_secret.key"

PBKDF2_ITERATIONS = 310_000          # OWASP推奨水準（PBKDF2-HMAC-SHA256）
SESSION_TTL_SECONDS = 8 * 60 * 60    # ログイン後の有効時間（8時間）
PENDING_TTL_SECONDS = 5 * 60         # パスワード通過後〜コード入力までの猶予（5分）
TOTP_STEP = 30
TOTP_DIGITS = 6
TOTP_ISSUER = "ACS FAQ Manager"
RECOVERY_CODE_COUNT = 8
MIN_PASSWORD_LENGTH = 10

MAX_FAILS_PER_USER = 5               # 同一ユーザーの連続失敗上限
MAX_FAILS_PER_IP = 20                # 同一IPの連続失敗上限
LOCK_SECONDS = 15 * 60               # ロック時間（15分）

_users_lock = threading.RLock()


# ============================================================
# パスワード
# ============================================================
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password_hash(stored: str, password: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ユーザーが存在しない場合でも照合にかかる時間を揃える（ユーザー名の推測対策）
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def check_password_policy(password: str, username: str = "") -> Optional[str]:
    """問題があればエラーメッセージ、問題なければNone。"""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"パスワードは{MIN_PASSWORD_LENGTH}文字以上にしてください"
    if username and password.lower() == username.lower():
        return "パスワードにユーザー名と同じ文字列は使えません"
    if password.lower() in ("admin", "password", "abc123", "12345678", "1234567890"):
        return "推測されやすいパスワードは使用できません"
    return None


# ============================================================
# TOTP（RFC 6238）
# ============================================================
def new_totp_secret() -> str:
    """Base32形式のシークレット（Authenticatorアプリが読める形式）"""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _hotp(secret_b32: str, counter: int) -> str:
    padded = secret_b32 + "=" * (-len(secret_b32) % 8)
    key = base64.b32decode(padded, casefold=True)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(code_int % (10 ** TOTP_DIGITS)).zfill(TOTP_DIGITS)


def verify_totp(secret_b32: str, code: str, last_used_step: int = 0, window: int = 1) -> Optional[int]:
    """
    コードが正しければ、そのコードの時間ステップ番号を返す（不正ならNone）。
    last_used_step 以下のステップは拒否する（同じコードの使い回し＝リプレイ攻撃対策）。
    window=1 は前後30秒のズレを許容する設定。
    """
    code = (code or "").strip().replace(" ", "")
    if not (code.isdigit() and len(code) == TOTP_DIGITS):
        return None
    now_step = int(time.time()) // TOTP_STEP
    for delta in range(-window, window + 1):
        step = now_step + delta
        if step <= last_used_step:
            continue
        if hmac.compare_digest(_hotp(secret_b32, step), code):
            return step
    return None


def totp_uri(secret_b32: str, username: str) -> str:
    label = quote(f"{TOTP_ISSUER}:{username}")
    return (
        f"otpauth://totp/{label}?secret={secret_b32}"
        f"&issuer={quote(TOTP_ISSUER)}&algorithm=SHA1&digits={TOTP_DIGITS}&period={TOTP_STEP}"
    )


def qr_svg(data: str) -> str:
    """QRコードをSVG文字列で返す。ライブラリ未導入時は空文字（手入力用の鍵表示で代替）。"""
    try:
        import io
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
        buf = io.BytesIO()
        img.save(buf)
        svg = buf.getvalue().decode("utf-8")
        # XML宣言を除去してHTMLに埋め込めるようにする
        return svg[svg.find("<svg"):]
    except Exception as e:
        print(f"⚠️ QR generation failed: {e}")
        return ""


# ============================================================
# リカバリーコード
# ============================================================
def _hash_recovery(code: str) -> str:
    return hashlib.sha256(code.strip().upper().replace("-", "").encode("utf-8")).hexdigest()


def generate_recovery_codes() -> Tuple[List[str], List[str]]:
    """(利用者に見せる平文コード, 保存用ハッシュ) を返す。形式: XXXXX-XXXXX"""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 紛らわしい文字(0,O,1,I)を除外
    plain, hashed = [], []
    for _ in range(RECOVERY_CODE_COUNT):
        raw = "".join(secrets.choice(alphabet) for _ in range(10))
        plain.append(f"{raw[:5]}-{raw[5:]}")
        hashed.append(_hash_recovery(raw))
    return plain, hashed


def consume_recovery_code(user: dict, code: str) -> bool:
    """リカバリーコードが一致すれば消費（使用済みに）してTrue。"""
    h = _hash_recovery(code)
    codes = user.get("recovery_codes", [])
    for stored in codes:
        if hmac.compare_digest(stored, h):
            codes.remove(stored)
            return True
    return False


# ============================================================
# 署名付きトークン
# ============================================================
def _signing_key() -> bytes:
    env = os.getenv("SESSION_SECRET", "")
    if env:
        return env.encode("utf-8")
    with _users_lock:
        if SESSION_KEY_PATH.exists():
            return SESSION_KEY_PATH.read_text(encoding="utf-8").strip().encode("utf-8")
        SESSION_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        SESSION_KEY_PATH.write_text(key, encoding="utf-8")
        try:
            os.chmod(SESSION_KEY_PATH, 0o600)
        except Exception:
            pass
        return key.encode("utf-8")


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(username: str, purpose: str, epoch: int, ttl: int) -> str:
    """purpose: 'session'(ログイン済み) / 'mfa'(コード入力待ち) / 'setup'(MFA登録待ち)"""
    payload = {"u": username, "p": purpose, "e": epoch, "x": int(time.time()) + ttl}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64e(hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def parse_token(token: str, purpose: str) -> Optional[dict]:
    """有効なら payload(dict)、無効・期限切れ・用途違いならNone。"""
    try:
        body, sig = token.split(".", 1)
        expected = _b64e(hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64d(body))
        if payload.get("p") != purpose or payload.get("x", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ============================================================
# ログイン失敗回数の制限（メモリ保持。再起動で解除される）
# ============================================================
_fail_state = {}  # key -> {"count": int, "locked_until": float}


def _fail_key(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def lock_remaining(username: str, ip: str) -> int:
    """ロック中なら残り秒数、ロックされていなければ0。"""
    now = time.time()
    remain = 0
    for k in (_fail_key("u", username.lower()), _fail_key("ip", ip)):
        st = _fail_state.get(k)
        if st and st["locked_until"] > now:
            remain = max(remain, int(st["locked_until"] - now))
    return remain


def record_failure(username: str, ip: str) -> None:
    now = time.time()
    for kind, value, limit in (("u", username.lower(), MAX_FAILS_PER_USER), ("ip", ip, MAX_FAILS_PER_IP)):
        k = _fail_key(kind, value)
        st = _fail_state.setdefault(k, {"count": 0, "locked_until": 0.0})
        if st["locked_until"] and st["locked_until"] <= now:
            st["count"], st["locked_until"] = 0, 0.0
        st["count"] += 1
        if st["count"] >= limit:
            st["locked_until"] = now + LOCK_SECONDS


def record_success(username: str) -> None:
    _fail_state.pop(_fail_key("u", username.lower()), None)


# ============================================================
# users.json の読み書き
# ============================================================
def load_users() -> dict:
    with _users_lock:
        if not USERS_PATH.exists():
            return {"users": []}
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("users", [])
        return data


def save_users(data: dict) -> None:
    """一時ファイルに書いてから置き換える（書き込み途中の破損を防ぐ）。"""
    with _users_lock:
        tmp = USERS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, USERS_PATH)
        try:
            os.chmod(USERS_PATH, 0o600)
        except Exception:
            pass


def find_user(data: dict, username: str) -> Optional[dict]:
    return next((u for u in data["users"] if u.get("username") == username), None)


def new_user_record(username: str, password: str, must_change_password: bool = True) -> dict:
    return {
        "username": username,
        "password_hash": hash_password(password),
        "mfa_enabled": False,
        "totp_secret": "",
        "pending_totp_secret": "",
        "totp_last_step": 0,
        "recovery_codes": [],
        "token_epoch": 1,
        "must_change_password": must_change_password,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def migrate_users_file() -> None:
    """
    起動時に1回実行。旧形式（平文パスワード・秘密の質問）のusers.jsonを新形式へ変換する。
      - password(平文) → password_hash（PBKDF2）に置換し、平文を削除
      - secret_question / secret_answer を削除（多要素認証をすり抜ける再設定手段のため廃止）
      - MFA関連の項目を補完（MFAは各管理者の次回ログイン時に登録）
    """
    with _users_lock:
        if not USERS_PATH.exists():
            return
        data = load_users()
        changed = False
        for u in data["users"]:
            if "password" in u:
                plain = u.pop("password")
                if "password_hash" not in u:
                    u["password_hash"] = hash_password(plain)
                    # 弱い既定パスワードのままなら、次回ログイン後に変更を促す
                    u.setdefault("must_change_password", check_password_policy(plain, u.get("username", "")) is not None)
                changed = True
            for legacy in ("secret_question", "secret_answer"):
                if legacy in u:
                    u.pop(legacy)
                    changed = True
            defaults = {
                "mfa_enabled": False, "totp_secret": "", "pending_totp_secret": "",
                "totp_last_step": 0, "recovery_codes": [], "token_epoch": 1,
                "must_change_password": False,
            }
            for k, v in defaults.items():
                if k not in u:
                    u[k] = v
                    changed = True
        if changed:
            save_users(data)
            print("🔐 users.json を新形式（ハッシュ化・MFA対応）へ移行しました")
