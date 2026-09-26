
  // ================================================================
  // 多言語対応
  // ================================================================
  
  var currentLanguage = localStorage.getItem('language') || 'ja';
  
  var translations = {
    ja: {
      headerSubtitle: 'サポート検索システム',
      heroTitle: 'サポート検索',
      heroSub: 'FAQ・サポート動画をまとめて検索できます',
      searchPlaceholder: 'キーワードを入力... (例: パスワード, 起動, 印刷)',
      emptyTitle: '検索してみましょう',
      emptyDesc: 'FAQ（よくある質問）とサポート動画を同時に検索します。<br>操作方法・トラブル対応など、気になるキーワードを入れてみましょう。',
      faqTitle: 'FAQ（よくある質問）',
      videoTitle: 'サポート動画',
      noFaqResult: '該当するFAQが見つかりませんでした。',
      noVideoResult: '該当する動画が見つかりませんでした。',
      faqPreparing: '準備中',
      prevPage: '前へ',
      nextPage: '次へ',
      close: '閉じる',
      watchOnYoutube: 'YouTubeで見る',
      count: '件'
    },
    en: {
      headerSubtitle: 'Support Search System',
      heroTitle: 'Support Search',
      heroSub: 'Search FAQ and support videos',
      searchPlaceholder: 'Enter keywords... (e.g., password, startup, print)',
      emptyTitle: 'Let's search',
      emptyDesc: 'Search FAQ and support videos simultaneously.<br>Enter keywords about operations, troubleshooting, etc.',
      faqTitle: 'FAQ (Frequently Asked Questions)',
      videoTitle: 'Support Videos',
      noFaqResult: 'No matching FAQ found.',
      noVideoResult: 'No matching videos found.',
      faqPreparing: 'Preparing',
      prevPage: 'Previous',
      nextPage: 'Next',
      close: 'Close',
      watchOnYoutube: 'Watch on YouTube',
      count: ' items'
    }
  };
  
  function switchLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    
    // ボタンのアクティブ状態を切り替え
    document.querySelectorAll('.lang-btn').forEach(function(btn) {
      if (btn.getAttribute('data-lang') === lang) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });
    
    // UIテキストを更新
    updateUIText();
    
    // 検索結果がある場合は再検索
    if (lastQuery) {
      doSearch(lastQuery);
    }
  }
  
  function updateUIText() {
    var t = translations[currentLanguage];
    
    document.getElementById('headerSubtitle').textContent = t.headerSubtitle;
    document.getElementById('heroTitle').textContent = t.heroTitle;
    document.getElementById('heroSub').textContent = t.heroSub;
    document.getElementById('searchInput').placeholder = t.searchPlaceholder;
    
    // 空の状態のテキスト
    var emptyTitle = document.querySelector('.empty-title');
    var emptyDesc = document.querySelector('.empty-desc');
    if (emptyTitle) emptyTitle.textContent = t.emptyTitle;
    if (emptyDesc) emptyDesc.innerHTML = t.emptyDesc;
  }
  
  function t(key) {
    return translations[currentLanguage][key] || key;
  }
  
  // ページ読み込み時に言語を適用
  document.addEventListener('DOMContentLoaded', function() {
    switchLanguage(currentLanguage);
  });
  
  // ================================================================
  // グローバル変数
  // ================================================================
  
  
  // ================================================================
  // 状態管理
  // ================================================================
  var PAGE_SIZE = 9;
  var lastQuery = '';
  var debounceTimer = null;
  var synonyms = {};
  var faqSearchEnabled = true; // FAQ検索の有効/無効

  var faqState = { offset: 0, items: [], total: 0 };
  var vidState = { offset: 0, items: [], total: 0 };

  // ================================================================
  // 初期化
  // ================================================================
  (function init() {
    // 設定を取得
    fetch('/api/config')
      .then(function(r) { return r.json(); })
      .then(function(config) {
        console.log('📋 Config loaded:', config);
        faqSearchEnabled = config.faq_search_enabled !== false;
        console.log('🔍 FAQ検索: ' + (faqSearchEnabled ? 'enabled' : 'disabled'));
        updateFaqSectionVisibility();
      })
      .catch(function(error) {
        console.error('❌ Config load error:', error);
        faqSearchEnabled = true;
        updateFaqSectionVisibility();
      });

    // 同義語辞書を取得（言語別構造対応）
    fetch('/api/synonyms')
      .then(function(r) { return r.json(); })
      .then(function(data) {
        // 新形式 {ja: {}, en: {}} から言語別に取得
        if (data && typeof data === 'object') {
          synonyms = data;
        } else {
          synonyms = { ja: {}, en: {} };
        }
      })
      .catch(function() { 
        synonyms = { ja: {}, en: {} };
      });

    // 検索入力のイベント
    var input = document.getElementById('searchInput');
    var clearBtn = document.getElementById('searchClear');

    input.addEventListener('input', function() {
      var val = input.value.trim();
      clearBtn.style.display = val ? 'block' : 'none';

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function() {
        doSearch(val);
      }, 300);
    });

    // Enterキーで即座に検索
    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        clearTimeout(debounceTimer);
        var val = input.value.trim();
        if (val) {
          doSearch(val);
        }
      }
    });

    clearBtn.addEventListener('click', function() {
      input.value = '';
      clearBtn.style.display = 'none';
      showEmpty();
      input.focus();
    });

    // Lightbox閉じる
    document.getElementById('faqLbClose').addEventListener('click', closeFaqLb);
    document.getElementById('vidLbClose').addEventListener('click', closeVidLb);

    document.getElementById('faqLightbox').addEventListener('click', function(e) {
      if (e.target === this) closeFaqLb();
    });
    document.getElementById('vidLightbox').addEventListener('click', function(e) {
      if (e.target === this) closeVidLb();
    });

    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') { closeFaqLb(); closeVidLb(); }
    });
  })();

  // ================================================================
  // 同義語展開（検索精度向上）
  // ================================================================
  function expandQuery(q) {
    if (!q || !synonyms) return q;
    
    // 言語別の同義語辞書を取得
    var langSynonyms = synonyms[currentLanguage] || {};
    
    var terms = [q];
    var qLower = q.toLowerCase();
    
    Object.keys(langSynonyms).forEach(function(term) {
      var syns = langSynonyms[term];
      if (!Array.isArray(syns)) return;
      
      // クエリに用語が含まれている場合、同義語を追加
      if (qLower.indexOf(term.toLowerCase()) !== -1) {
        syns.forEach(function(s) {
          if (terms.indexOf(s) === -1) terms.push(s);
        });
      }
      
      // クエリに同義語が含まれている場合、元の用語を追加
      syns.forEach(function(s) {
        if (qLower.indexOf(s.toLowerCase()) !== -1) {
          if (terms.indexOf(term) === -1) terms.push(term);
        }
      });
    });
    
    return terms.join(' ');
  }

  // ================================================================
  // 検索実行
  // ================================================================
  function doSearch(q) {
    console.log('🔍 Search started:', q);
    if (!q) { showEmpty(); return; }
    if (q === lastQuery) return;
    lastQuery = q;

    faqState = { offset: 0, items: [], total: 0 };
    vidState = { offset: 0, items: [], total: 0 };

    showLoading();

    var expanded = expandQuery(q);
    console.log('📝 Expanded query:', expanded);
    console.log('🌐 Language:', currentLanguage);
    console.log('📚 FAQ enabled:', faqSearchEnabled);

    // FAQ検索がOFFの場合はスキップ
    var faqPromise;
    if (faqSearchEnabled) {
      faqPromise = fetch('/faq/search?query=' + encodeURIComponent(expanded) + '&offset=0&limit=50&paged=1&language=' + currentLanguage)
        .then(function(r) { return r.json(); })
        .catch(function() { return { items: [], total_visible: 0 }; });
    } else {
      faqPromise = Promise.resolve({ items: [], total_visible: 0 });
    }

    var vidPromise = fetch('/search?query=' + encodeURIComponent(expanded) + '&offset=0&limit=50&paged=1&language=' + currentLanguage)
      .then(function(r) { return r.json(); })
      .catch(function() { return { items: [], total_visible: 0 }; });

    Promise.all([faqPromise, vidPromise]).then(function(results) {
      var faqRes = results[0];
      var vidRes = results[1];

      console.log('✅ Search results received');
      console.log('  FAQ:', faqRes.items ? faqRes.items.length : 0, 'items');
      console.log('  Videos:', vidRes.items ? vidRes.items.length : 0, 'items');

      faqState.items = faqRes.items || [];
      faqState.total = faqRes.total_visible || faqState.items.length;

      vidState.items = vidRes.items || [];
      vidState.total = vidRes.total_visible || vidState.items.length;

      renderResults();
    }).catch(function(error) {
      console.error('❌ Search error:', error);
      showEmpty();
    });
  }

  // ================================================================
  // 描画
  // ================================================================
  function renderResults() {
    showResults();
    renderFaq();
    renderVideos();
  }

  function renderFaq() {
    var list = document.getElementById('faqList');
    var countEl = document.getElementById('faqCount');
    var pagination = document.getElementById('faqPagination');
    var total = faqState.total;

    countEl.textContent = total + '件';

    var start = faqState.offset;
    var end = Math.min(start + PAGE_SIZE, faqState.items.length);
    var pageItems = faqState.items.slice(start, end);

    if (pageItems.length === 0) {
      // FAQ検索がONでかつ全件数が0の場合は「準備中」を表示
      if (faqSearchEnabled && faqState.total === 0) {
        list.innerHTML = '<div class="faq-preparing">準備中</div>';
      } else {
        list.innerHTML = '<div class="no-result">該当するFAQが見つかりませんでした。</div>';
      }
      pagination.style.display = 'none';
      return;
    }

    list.innerHTML = '';
    pageItems.forEach(function(item, i) {
      var card = document.createElement('div');
      card.className = 'faq-card';
      card.innerHTML =
        '<div class="faq-category">' + esc(item.category || 'その他') + '</div>' +
        '<div class="faq-question">' + esc(item.question || '') + '</div>';
      card.addEventListener('click', function() { 
        openFaqLb(item); 
        if (lastQuery) {
          logClick(lastQuery, 'faq', item.id);
        }
      });
      list.appendChild(card);
    });

    // ページネーション
    var totalPages = Math.ceil(faqState.items.length / PAGE_SIZE);
    var currentPage = Math.floor(faqState.offset / PAGE_SIZE) + 1;

    if (totalPages > 1) {
      pagination.style.display = 'flex';
      document.getElementById('faqPrev').disabled = faqState.offset === 0;
      document.getElementById('faqNext').disabled = end >= faqState.items.length;
      document.getElementById('faqPageInfo').textContent = currentPage + ' / ' + totalPages + ' ページ';
    } else {
      pagination.style.display = 'none';
    }
  }

  function renderVideos() {
    var grid = document.getElementById('videoGrid');
    var countEl = document.getElementById('vidCount');
    var pagination = document.getElementById('vidPagination');
    var total = vidState.total;

    countEl.textContent = total + '件';

    var start = vidState.offset;
    var end = Math.min(start + PAGE_SIZE, vidState.items.length);
    var pageItems = vidState.items.slice(start, end);

    if (pageItems.length === 0) {
      grid.innerHTML = '<div class="no-result" style="grid-column:1/-1;">該当する動画が見つかりませんでした。</div>';
      pagination.style.display = 'none';
      return;
    }

    grid.innerHTML = '';
    pageItems.forEach(function(item) {
      var vid = getVideoId(item);
      var thumb = item.thumbnail || (vid ? 'https://i.ytimg.com/vi/' + vid + '/mqdefault.jpg' : '');

      var card = document.createElement('div');
      card.className = 'video-card';
      card.innerHTML =
        '<div class="video-thumb-wrap">' +
          (thumb ? '<img class="video-thumb" src="' + esc(thumb) + '" alt="' + esc(item.title || '') + '" loading="lazy" />' : '<div class="video-thumb" style="background:#333;position:absolute;inset:0;"></div>') +
          '<div class="video-play-btn">▶</div>' +
        '</div>' +
        '<div class="video-info"><div class="video-title">' + esc(item.title || '') + '</div></div>';

      card.addEventListener('click', function() { 
        openVidLb(item, vid); 
        if (lastQuery) {
          logClick(lastQuery, 'video', item.video_id || '');
        }
      });
      grid.appendChild(card);
    });

    // ページネーション
    var totalPages = Math.ceil(vidState.items.length / PAGE_SIZE);
    var currentPage = Math.floor(vidState.offset / PAGE_SIZE) + 1;

    if (totalPages > 1) {
      pagination.style.display = 'flex';
      document.getElementById('vidPrev').disabled = vidState.offset === 0;
      document.getElementById('vidNext').disabled = end >= vidState.items.length;
      document.getElementById('vidPageInfo').textContent = currentPage + ' / ' + totalPages + ' ページ';
    } else {
      pagination.style.display = 'none';
    }
  }

  // ================================================================
  // ページング
  // ================================================================
  function changePage(type, dir) {
    if (type === 'faq') {
      faqState.offset = Math.max(0, faqState.offset + dir * PAGE_SIZE);
      renderFaq();
      document.getElementById('faqList').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      vidState.offset = Math.max(0, vidState.offset + dir * PAGE_SIZE);
      renderVideos();
      document.getElementById('videoGrid').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // ================================================================
  // Lightbox - FAQ（Bug#2修正: stepsを番号付きリストで表示）
  // ================================================================
  function openFaqLb(item) {
    document.getElementById('lbCategory').textContent = item.category || 'その他';
    document.getElementById('lbQuestion').textContent = item.question || '';

    // 回答の表示
    var answerSection = document.getElementById('lbAnswerSection');
    var answerText = document.getElementById('lbAnswerText');
    if (item.answer && item.answer.trim()) {
      answerText.textContent = item.answer;
      answerSection.style.display = 'block';
    } else {
      answerSection.style.display = 'none';
    }

    // 手順の表示
    var steps = item.steps || [];
    var stepsEl = document.getElementById('lbSteps');
    stepsEl.innerHTML = '';
    if (steps.length === 0) {
      stepsEl.innerHTML = '<li><span>詳細情報がありません。</span></li>';
    } else {
      steps.forEach(function(s) {
        var li = document.createElement('li');
        li.innerHTML = '<span>' + esc(s) + '</span>';
        stepsEl.appendChild(li);
      });
    }

    // ワンポイントアドバイスの表示
    var noteSection = document.getElementById('lbNoteSection');
    var noteText = document.getElementById('lbNoteText');
    if (item.note && item.note.trim()) {
      noteText.textContent = item.note;
      noteSection.style.display = 'block';
    } else {
      noteSection.style.display = 'none';
    }

    document.getElementById('faqLightbox').classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  
  function closeFaqLb() {
    document.getElementById('faqLightbox').classList.remove('active');
    document.body.style.overflow = '';
  }

  // ================================================================
  // Lightbox - 動画（Bug#3修正: allow属性追加、Bug#6: video_id取得）
  // ================================================================
  function openVidLb(item, vid) {
    document.getElementById('lbVideoTitle').textContent = item.title || '';
    document.getElementById('lbVideoDesc').textContent = item.description || '';

    var wrapEl = document.getElementById('lbIframeWrap');
    wrapEl.innerHTML = '';

    if (vid) {
      var iframe = document.createElement('iframe');
      iframe.src = 'https://www.youtube.com/embed/' + vid + '?autoplay=1&rel=0';
      // Bug#3修正: allow属性を正しく設定
      iframe.setAttribute('allow', 'autoplay; encrypted-media; fullscreen; picture-in-picture');
      iframe.setAttribute('allowfullscreen', '');
      iframe.setAttribute('frameborder', '0');
      iframe.title = item.title || '';
      wrapEl.appendChild(iframe);
    } else {
      wrapEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#fff;background:#222;padding:40px;text-align:center;">動画IDが取得できませんでした。</div>';
    }

    document.getElementById('vidLightbox').classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeVidLb() {
    // Bug#3修正: 閉じたときにiframeをクリアして動画停止
    document.getElementById('lbIframeWrap').innerHTML = '';
    document.getElementById('vidLightbox').classList.remove('active');
    document.body.style.overflow = '';
  }

  // ================================================================
  // Bug#6修正: video_id取得（フォールバック付き）
  // ================================================================
  function getVideoId(item) {
    var vid = String(item.video_id || '').trim();
    if (vid) return vid;
    var url = String(item.url || '');
    var m = url.match(/[?&]v=([A-Za-z0-9_-]{11})/);
    if (m) return m[1];
    var m2 = url.match(/youtu\.be\/([A-Za-z0-9_-]{11})/);
    if (m2) return m2[1];
    return '';
  }

  // ================================================================
  // ログ記録
  // ================================================================
  function logClick(query, resultType, resultId) {
    fetch('/api/log_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        result_type: resultType,
        result_id: resultId,
        timestamp: new Date().toISOString()
      })
    }).catch(function() {});
  }

  // ================================================================
  // UI状態切り替え
  // ================================================================
  // FAQ検索の有効/無効に応じて表示を切り替え
  function updateFaqSectionVisibility() {
    var faqSection = document.getElementById('faqSection');
    var videoSection = document.getElementById('videoSection');
    var resultsGrid = document.querySelector('.results-grid');
    var heroSub = document.getElementById('heroSub');
    var emptyStateDesc = document.getElementById('emptyStateDesc');
    
    if (faqSearchEnabled) {
      // FAQ検索ON: 通常の2カラム表示
      faqSection.style.display = '';
      resultsGrid.classList.remove('video-only');
      
      // 文言を変更
      heroSub.textContent = 'FAQ・サポート動画をまとめて検索できます';
      emptyStateDesc.innerHTML = 'FAQ（よくある質問）とサポート動画を同時に検索します。<br>操作方法・トラブル対応など、気になるキーワードを入れてみましょう。';
    } else {
      // FAQ検索OFF: FAQセクションを非表示、動画のみ1カラム表示
      faqSection.style.display = 'none';
      resultsGrid.classList.add('video-only');
      
      // 文言を変更
      heroSub.textContent = 'サポート動画を検索します';
      emptyStateDesc.innerHTML = 'サポート動画を検索します<br>操作方法・トラブル対応など、気になるキーワードを入れてみましょう。';
    }
  }

    function showEmpty() {
    lastQuery = '';
    document.getElementById('emptyState').style.display = '';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('resultsArea').style.display = 'none';
  }

  function showLoading() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingState').style.display = '';
    document.getElementById('resultsArea').style.display = 'none';
  }

  function showResults() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('resultsArea').style.display = '';
  }

  // ================================================================
  // エスケープ
  // ================================================================
  function esc(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  