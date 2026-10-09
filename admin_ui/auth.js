/*
 * 管理画面 共通認証スクリプト（ログイン画面以外の全ページで読み込む）
 *  - ログイン情報が旧形式/未設定ならログイン画面へ戻す
 *  - APIが401（ログイン期限切れ・パスワード変更・MFAリセット等）を返したらログイン画面へ戻す
 *  - 初期パスワードのままの管理者には、パスワード変更を促すバナーを表示する
 */
(function () {
  var LOGIN_URL = '/admin';

  function backToLogin(message) {
    try { localStorage.removeItem('admin_auth'); } catch (e) {}
    if (message) { try { sessionStorage.setItem('login_notice', message); } catch (e) {} }
    window.location.href = LOGIN_URL;
  }

  // 1) ログイン情報の確認（トークンが無い＝未ログイン or 旧方式のログイン情報）
  var raw = null, auth = null;
  try { raw = localStorage.getItem('admin_auth'); auth = raw ? JSON.parse(raw) : null; } catch (e) {}
  if (!auth || !auth.token) { backToLogin(''); return; }

  // 2) 401 を検知したらログイン画面へ（ログインAPI自身は除く）
  var origFetch = window.fetch;
  window.fetch = function (input, init) {
    return origFetch.apply(this, arguments).then(function (res) {
      var url = typeof input === 'string' ? input : (input && input.url) || '';
      if (res.status === 401 && url.indexOf('/admin/api/') !== -1 && url.indexOf('/admin/api/auth/') === -1) {
        backToLogin('ログインの有効期限が切れました。もう一度ログインしてください。');
      }
      return res;
    });
  };

  // ナビのリンク数が増えたため、狭い画面でも全リンクが収まるよう余白を詰める
  var st = document.createElement('style');
  st.textContent = 'nav { min-width: 0; } nav a { padding: 7px 8px !important; font-size: 12.5px !important; }';
  document.head.appendChild(st);

  // 3) パスワード変更を促すバナー
  document.addEventListener('DOMContentLoaded', function () {
    if (location.pathname.indexOf('/admin/password') === 0) return;
    fetch('/admin/api/user', { headers: { 'Authorization': 'Bearer ' + auth.token } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (u) {
        if (!u || !u.must_change_password) return;
        var bar = document.createElement('div');
        bar.style.cssText = 'background:#fffbeb;border-bottom:1px solid #f59e0b;color:#92400e;padding:10px 24px;font-size:13px;text-align:center;';
        bar.innerHTML = '⚠ 初期（または一時）パスワードのままです。 <a href="/admin/password" style="color:#92400e;font-weight:700;">パスワードを変更してください</a>';
        document.body.insertBefore(bar, document.body.firstChild);
      })
      .catch(function () {});
  });
})();
