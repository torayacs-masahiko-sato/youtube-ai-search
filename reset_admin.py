"""
管理者アカウント緊急復旧ツール（Render の Shell から実行）

使い方:
  cd /opt/render/project/src
  python reset_admin.py <ユーザー名> <新しい一時パスワード>

動作:
  - 指定ユーザーのパスワードを一時パスワードに変更し、多要素認証(MFA)を解除する
  - ユーザーが存在しなければ新規に管理者として作成する
  - 次回ログイン時に、MFAの再登録とパスワード変更が求められる

使いどころ:
  管理者全員がログインできなくなった場合の最終手段。
  （他の管理者がいる場合は、管理画面「管理者管理」から再設定できる）
前提:
  環境変数 DATA_DIR が本番と同じであること（Render Shell では自動で設定済み）
"""
import sys
import auth_utils as au

def main():
    if len(sys.argv) != 3:
        print("使い方: python reset_admin.py <ユーザー名> <新しい一時パスワード>")
        sys.exit(1)
    username, password = sys.argv[1], sys.argv[2]
    problem = au.check_password_policy(password, username)
    if problem:
        print(f"エラー: {problem}")
        sys.exit(1)

    data = au.load_users()
    user = au.find_user(data, username)
    if user is None:
        data["users"].append(au.new_user_record(username, password, must_change_password=True))
        print(f"管理者 '{username}' を新規作成しました")
    else:
        user.update({
            "password_hash": au.hash_password(password), "must_change_password": True,
            "mfa_enabled": False, "totp_secret": "", "pending_totp_secret": "",
            "totp_last_step": 0, "recovery_codes": [],
            "token_epoch": user.get("token_epoch", 1) + 1,
        })
        print(f"管理者 '{username}' のパスワードを再設定し、MFAを解除しました")
    au.save_users(data)
    print("次回ログイン時にMFAの再登録とパスワード変更を行ってください")

if __name__ == "__main__":
    main()
