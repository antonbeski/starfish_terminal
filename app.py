import re
import time
import traceback
import requests
from flask import Flask, request, jsonify
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd

app = Flask(__name__)

# ── Live news channel YouTube handles ───────────────────────────────────────
NEWS_CHANNELS = [
    {"id": "awaaz",      "handle": "CNBCAwaaz",        "label": "CNBC Awaaz",   "lang": "HI", "region": "India"},
    {"id": "zeebusiness","handle": "ZeeBusiness",      "label": "Zee Business", "lang": "HI", "region": "India"},
    {"id": "bloomberg",  "handle": "Bloomberg",        "label": "Bloomberg TV", "lang": "EN", "region": "Global"},
    {"id": "aljazeera",  "handle": "aljazeeraenglish", "label": "Al Jazeera",   "lang": "EN", "region": "Global"},
]

_news_cache = {}   # {handle: {"video_id": str, "ts": float}}
_NEWS_TTL   = 3600  # re-fetch after 1 hour

_YT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_live_video_id(handle: str) -> str | None:
    """Scrape the channel's /live page and extract the current live video ID."""
    cached = _news_cache.get(handle)
    if cached and time.time() - cached["ts"] < _NEWS_TTL:
        return cached["video_id"]

    url = f"https://www.youtube.com/@{handle}/live"
    try:
        r = requests.get(url, headers=_YT_HEADERS, timeout=10, allow_redirects=True)
        # YouTube redirects /live to the actual watch URL when live
        # The final URL contains the video ID
        final_url = r.url  # e.g. https://www.youtube.com/watch?v=XXXXXXXXXXX
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", final_url)
        if not m:
            # Also try scraping from page HTML
            m = re.search(r'"videoId":"([A-Za-z0-9_-]{11})"', r.text)
        video_id = m.group(1) if m else None
        _news_cache[handle] = {"video_id": video_id, "ts": time.time()}
        return video_id
    except Exception:
        return None

POPULAR_STOCKS = [
    ("AAPL", "Apple"), ("GOOGL", "Google"), ("MSFT", "Microsoft"),
    ("TSLA", "Tesla"), ("AMZN", "Amazon"), ("NVDA", "NVIDIA"),
    ("TCS.NS", "TCS"), ("RELIANCE.NS", "Reliance"),
]

PERIODS = [
    ("1mo", "1 Month"), ("3mo", "3 Months"), ("6mo", "6 Months"),
    ("1y", "1 Year"), ("2y", "2 Years"), ("5y", "5 Years"),
]
VALID_PERIODS = {p[0] for p in PERIODS}

# ── Yahoo Finance cookie + crumb setup ──────────────────────────────────────
# Yahoo requires a valid session cookie and a "crumb" token before it will
# serve any data. On Vercel, yfinance's auto-fetch for these fails silently.
# We fetch them manually and inject them into yfinance's session.

_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://finance.yahoo.com/",
    "Origin": "https://finance.yahoo.com",
    "DNT": "1",
    "Connection": "keep-alive",
}

_session_cache = {}   # simple in-process cache so we don't re-fetch every request


def get_yahoo_session():
    """
    Return a requests.Session with a valid Yahoo Finance cookie + crumb.
    Tries multiple Yahoo endpoints in order until one succeeds.
    """
    if _session_cache.get("session"):
        return _session_cache["session"], _session_cache.get("crumb")

    session = requests.Session()
    session.headers.update(_YF_HEADERS)

    # Step 1: hit the consent / main page to get cookies
    cookie_urls = [
        "https://fc.yahoo.com",
        "https://finance.yahoo.com/",
        "https://login.yahoo.com/",
    ]
    for url in cookie_urls:
        try:
            session.get(url, timeout=10, allow_redirects=True)
            break
        except Exception:
            continue

    # Step 2: fetch the crumb
    crumb = None
    crumb_urls = [
        "https://query1.finance.yahoo.com/v1/test/getcrumb",
        "https://query2.finance.yahoo.com/v1/test/getcrumb",
    ]
    for url in crumb_urls:
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200 and r.text and "<" not in r.text:
                crumb = r.text.strip()
                break
        except Exception:
            continue

    _session_cache["session"] = session
    _session_cache["crumb"] = crumb
    return session, crumb


def fetch_yfinance_data(ticker, period):
    """
    Download ticker data using a manually obtained Yahoo Finance session.
    Falls back to direct yfinance download if the manual approach fails.
    """
    session, crumb = get_yahoo_session()

    # Inject our session into yfinance
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            actions=False,
            session=session,
        )
        if data is not None and not data.empty:
            return data, None
    except Exception as e:
        pass

    # Fallback: fresh session, no crumb
    try:
        _session_cache.clear()
        fresh_session = requests.Session()
        fresh_session.headers.update(_YF_HEADERS)
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            actions=False,
            session=fresh_session,
        )
        if data is not None and not data.empty:
            return data, None
    except Exception as e:
        return None, str(e)

    return None, None


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_chart(ticker, period, chart_type):
    data, err = fetch_yfinance_data(ticker, period)

    if err:
        return None, f"Download failed: {err}"

    if data is None or data.empty:
        return None, (
            f"No data found for '{ticker}'. "
            "Check the symbol — use '.NS' for NSE stocks (e.g. TCS.NS)."
        )

    data = flatten_columns(data)
    missing = {"Open", "High", "Low", "Close"} - set(data.columns)
    if missing:
        return None, f"Missing columns: {missing}"

    data = data.dropna(subset=["Close"])
    if data.empty:
        return None, "All rows were empty after cleaning."

    try:
        session, _ = get_yahoo_session()
        t = yf.Ticker(ticker, session=session)
        name = t.fast_info.get("longName") or t.info.get("shortName") or ticker
    except Exception:
        name = ticker

    currency = "INR" if ticker.upper().endswith(".NS") else "USD"

    if chart_type == "candlestick":
        trace = go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
            increasing_line_color="#ffffff",
            decreasing_line_color="#555555",
            increasing_fillcolor="rgba(255,255,255,0.15)",
            decreasing_fillcolor="rgba(80,80,80,0.15)",
        )
    else:
        trace = go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=ticker,
            line=dict(color="#ffffff", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,255,255,0.04)",
        )

    fig = go.Figure(data=trace)
    fig.update_layout(
        title=dict(
            text=f"{name} ({ticker.upper()})",
            font=dict(size=18, color="#ffffff", family="'DM Sans', sans-serif"),
        ),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#888888", family="'DM Sans', sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            rangeslider=dict(visible=False),
            color="#666666",
            showline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            color="#666666",
            showline=False,
        ),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(
            bgcolor="rgba(15,15,15,0.95)",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(color="#ffffff"),
        ),
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False), None


def render_page(ticker, period, chart_type, graph_html, error):
    chips = ""
    for sym, _ in POPULAR_STOCKS:
        active = "chip active" if sym == ticker else "chip"
        chips += f'<span class="{active}" onclick="setTicker(\'{sym}\')">{sym}</span>\n'

    period_opts = ""
    for val, label in PERIODS:
        sel = "selected" if val == period else ""
        period_opts += f'<option value="{val}" {sel}>{label}</option>\n'

    ct_candle = "selected" if chart_type == "candlestick" else ""
    ct_line   = "selected" if chart_type == "line" else ""

    if error:
        content = f'<div class="error-box">{error}</div>'
    elif graph_html:
        content = graph_html
    else:
        content = '<div class="empty-state">Enter a ticker symbol above to load a chart.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Starfish</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:          #060606;
      --surface:     rgba(255,255,255,0.04);
      --border:      rgba(255,255,255,0.09);
      --border-soft: rgba(255,255,255,0.05);
      --text:        #f0f0f0;
      --text-muted:  #666666;
      --text-dim:    #3a3a3a;
      --accent:      #ffffff;
      --accent-mute: rgba(255,255,255,0.1);
      --blur:        blur(20px);
      --radius:      16px;
      --radius-sm:   9px;
    }}

    body {{
      font-family: 'DM Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
    }}

    body::before {{
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 90% 55% at 15% 5%,  rgba(255,255,255,0.022) 0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 85% 85%, rgba(255,255,255,0.012) 0%, transparent 50%);
      pointer-events: none;
      z-index: 0;
    }}

    header {{
      position: sticky;
      top: 0;
      z-index: 100;
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      background: rgba(6,6,6,0.75);
      backdrop-filter: var(--blur);
      -webkit-backdrop-filter: var(--blur);
      border-bottom: 1px solid var(--border-soft);
    }}

    .logo {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
    }}

    .logo-pip {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: blink 2.8s ease-in-out infinite;
      flex-shrink: 0;
    }}

    @keyframes blink {{
      0%,100% {{ opacity: 1; transform: scale(1); }}
      50%      {{ opacity: 0.2; transform: scale(0.65); }}
    }}

    .subtitle {{
      font-size: 0.72rem;
      color: var(--text-dim);
      letter-spacing: 0.03em;
    }}

    main {{
      position: relative;
      z-index: 1;
      max-width: 1160px;
      margin: 0 auto;
      padding: 30px 20px 64px;
    }}

    .glass {{
      background: var(--surface);
      backdrop-filter: var(--blur);
      -webkit-backdrop-filter: var(--blur);
      border: 1px solid var(--border);
      border-radius: var(--radius);
    }}

    .panel {{
      padding: 26px 30px;
      margin-bottom: 18px;
    }}

    .panel-label {{
      font-size: 0.62rem;
      font-weight: 600;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 20px;
    }}

    form {{
      display: grid;
      grid-template-columns: 1.5fr 1fr 1fr auto;
      gap: 14px;
      align-items: end;
    }}

    .field-group label {{
      display: block;
      font-size: 0.7rem;
      font-weight: 500;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}

    input, select {{
      width: 100%;
      background: rgba(255,255,255,0.035);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      color: var(--text);
      padding: 10px 14px;
      font-size: 0.875rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
      appearance: none;
      -webkit-appearance: none;
    }}

    input::placeholder {{ color: var(--text-dim); }}

    input:focus, select:focus {{
      border-color: rgba(255,255,255,0.28);
      background: rgba(255,255,255,0.065);
      box-shadow: 0 0 0 3px rgba(255,255,255,0.05);
    }}

    select {{
      cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 13px center;
      padding-right: 34px;
    }}

    select option {{ background: #111111; color: #f0f0f0; }}

    .btn {{
      background: var(--accent);
      color: #000000;
      border: none;
      border-radius: var(--radius-sm);
      padding: 10px 26px;
      font-size: 0.8rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      white-space: nowrap;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      transition: opacity 0.18s, transform 0.13s;
      height: 42px;
    }}

    .btn:hover  {{ opacity: 0.85; }}
    .btn:active {{ transform: scale(0.96); }}

    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 22px;
      padding-top: 20px;
      border-top: 1px solid var(--border-soft);
    }}

    .chip {{
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 100px;
      padding: 5px 15px;
      font-size: 0.72rem;
      font-family: 'DM Mono', monospace;
      cursor: pointer;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      transition: all 0.16s;
      user-select: none;
    }}

    .chip:hover {{
      border-color: rgba(255,255,255,0.3);
      color: var(--text);
      background: var(--accent-mute);
    }}

    .chip.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #000000;
      font-weight: 600;
    }}

    .chart-card {{
      padding: 24px 20px 16px;
      min-height: 460px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}

    .chart-card > div {{ width: 100%; }}

    .error-box {{
      border: 1px solid rgba(255,255,255,0.1);
      border-left: 3px solid rgba(255,255,255,0.45);
      border-radius: var(--radius-sm);
      padding: 16px 20px;
      color: #999999;
      font-size: 0.875rem;
      background: rgba(255,255,255,0.025);
      width: 100%;
      line-height: 1.6;
    }}

    .empty-state {{
      color: var(--text-dim);
      font-size: 0.85rem;
      text-align: center;
      letter-spacing: 0.03em;
    }}

    /* ── Live News Panel ── */
    .news-panel {{
      padding: 26px 30px;
      margin-top: 18px;
    }}

    .news-live-dot {{
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #ff4444;
      margin-right: 6px;
      animation: livepulse 1.4s ease-in-out infinite;
      vertical-align: middle;
    }}

    @keyframes livepulse {{
      0%,100% {{ opacity: 1; transform: scale(1); }}
      50%      {{ opacity: 0.3; transform: scale(0.6); }}
    }}

    .news-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}

    .news-tab {{
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 100px;
      padding: 6px 18px;
      font-size: 0.72rem;
      font-family: 'DM Mono', monospace;
      cursor: pointer;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      transition: all 0.16s;
      user-select: none;
    }}

    .news-tab:hover {{
      border-color: rgba(255,255,255,0.3);
      color: var(--text);
      background: var(--accent-mute);
    }}

    .news-tab.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: #000000;
      font-weight: 600;
    }}

    .news-iframe-wrap {{
      position: relative;
      width: 100%;
      padding-top: 56.25%;
      border-radius: var(--radius-sm);
      overflow: hidden;
      background: rgba(0,0,0,0.5);
    }}

    .news-loading {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-muted);
      font-size: 0.8rem;
      letter-spacing: 0.05em;
    }}

    .news-iframe-wrap iframe {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      border: none;
    }}

    .news-notice {{
      margin-top: 12px;
      font-size: 0.66rem;
      color: var(--text-dim);
      letter-spacing: 0.03em;
    }}

    .news-tag {{
      font-size: 0.55rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 1px 5px;
      border-radius: 4px;
      background: rgba(255,255,255,0.08);
      color: var(--text-dim);
      margin-left: 4px;
      vertical-align: middle;
    }}

    .news-tab.active .news-tag {{
      background: rgba(0,0,0,0.15);
      color: rgba(0,0,0,0.5);
    }}

    @media (max-width: 860px) {{
      form {{ grid-template-columns: 1fr 1fr; gap: 12px; }}
      .field-group:first-child {{ grid-column: span 2; }}
      .btn {{ grid-column: span 2; width: 100%; }}
    }}

    @media (max-width: 600px) {{
      header {{ padding: 0 16px; }}
      .subtitle {{ display: none; }}
      main {{ padding: 18px 14px 48px; }}
      .panel {{ padding: 20px 18px; }}
      .chart-card {{ padding: 16px 10px 10px; min-height: 300px; }}
      .news-panel {{ padding: 20px 18px; }}
    }}

    @media (max-width: 400px) {{
      form {{ grid-template-columns: 1fr; }}
      .field-group:first-child {{ grid-column: span 1; }}
      .btn {{ grid-column: span 1; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="logo">
    <span class="logo-pip"></span>
    Starfish
  </div>
  <span class="subtitle">US &amp; NSE markets</span>
</header>

<main>
  <div class="glass panel">
    <div class="panel-label">Search</div>
    <form method="POST" action="/">
      <div class="field-group">
        <label for="ticker">Ticker Symbol</label>
        <input id="ticker" name="ticker" type="text" value="{ticker}"
               placeholder="AAPL, GOOGL, TCS.NS" required
               autocomplete="off" autocapitalize="characters" spellcheck="false"/>
      </div>
      <div class="field-group">
        <label for="period">Time Range</label>
        <select id="period" name="period">{period_opts}</select>
      </div>
      <div class="field-group">
        <label for="chart_type">Chart Type</label>
        <select id="chart_type" name="chart_type">
          <option value="candlestick" {ct_candle}>Candlestick</option>
          <option value="line" {ct_line}>Line</option>
        </select>
      </div>
      <button type="submit" class="btn">Search</button>
    </form>
    <div class="chips">{chips}</div>
  </div>

  <div class="glass chart-card">{content}</div>

  <!-- ── Live News Panel ── -->
  <div class="glass news-panel">
    <div class="panel-label">
      <span class="news-live-dot"></span>Live Financial News
    </div>
    <div class="news-tabs" id="news-tabs">
      <button class="news-tab active" data-handle="CNBCAwaaz">CNBC Awaaz <span class="news-tag">India</span></button>
      <button class="news-tab" data-handle="ZeeBusiness">Zee Business <span class="news-tag">India</span></button>
      <button class="news-tab" data-handle="Bloomberg">Bloomberg TV <span class="news-tag">Global</span></button>
      <button class="news-tab" data-handle="aljazeeraenglish">Al Jazeera <span class="news-tag">Global</span></button>
    </div>
    <div class="news-iframe-wrap">
      <div id="news-loading" class="news-loading">Loading live stream&#8230;</div>
      <iframe id="news-iframe"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen style="display:none">
      </iframe>
    </div>
    <p class="news-notice">&#x25CF; Active during market hours (9:15 AM – 3:30 PM IST). If a channel is off-air the video will show as unavailable.</p>
  </div>
</main>

<script>
  function setTicker(s) {{
    document.getElementById('ticker').value = s;
    document.querySelector('form').submit();
  }}

  // ── Live news player ──
  const iframe   = document.getElementById('news-iframe');
  const loading  = document.getElementById('news-loading');

  function loadChannel(handle) {{
    iframe.style.display = 'none';
    loading.style.display = 'flex';
    loading.textContent = 'Loading live stream\u2026';

    fetch('/api/live-id?handle=' + encodeURIComponent(handle))
      .then(r => r.json())
      .then(data => {{
        if (data.video_id) {{
          iframe.src = 'https://www.youtube.com/embed/' + data.video_id + '?autoplay=1&rel=0';
          iframe.style.display = 'block';
          loading.style.display = 'none';
        }} else {{
          loading.textContent = 'Stream unavailable right now. Try another channel.';
        }}
      }})
      .catch(() => {{
        loading.textContent = 'Could not load stream. Check your connection.';
      }});
  }}

  document.getElementById('news-tabs').addEventListener('click', function(e) {{
    const btn = e.target.closest('.news-tab');
    if (!btn) return;
    document.querySelectorAll('.news-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    loadChannel(btn.dataset.handle);
  }});

  // Auto-load first channel on page load
  loadChannel('CNBCAwaaz');
</script>
</body>
</html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    ticker     = (request.form.get("ticker", "AAPL") or "AAPL").strip().upper()
    period     = request.form.get("period", "6mo")
    chart_type = request.form.get("chart_type", "candlestick")

    if period not in VALID_PERIODS:
        period = "6mo"
    if chart_type not in ("candlestick", "line"):
        chart_type = "candlestick"

    graph_html, error = build_chart(ticker, period, chart_type)
    return render_page(ticker, period, chart_type, graph_html, error)


@app.route("/api/live-id")
def api_live_id():
    """Return the current YouTube live video ID for a given channel handle."""
    handle = request.args.get("handle", "")
    if not handle:
        return jsonify({"error": "missing handle"}), 400
    video_id = fetch_live_video_id(handle)
    if video_id:
        return jsonify({"video_id": video_id})
    return jsonify({"error": "not found"}), 404


@app.route("/debug")
def debug():
    out = []
    try:
        session, crumb = get_yahoo_session()
        out.append(f"Cookies: {dict(session.cookies)}")
        out.append(f"Crumb:   {crumb!r}")
        data = yf.download("AAPL", period="5d", progress=False,
                           auto_adjust=True, session=session)
        out.append(f"Shape:   {data.shape}")
        out.append(data.tail().to_string())
        color = "#7fff7f"
    except Exception:
        out.append(traceback.format_exc())
        color = "#ff7f7f"

    body = "\n".join(out)
    return (
        f"<pre style='background:#111;color:{color};padding:24px;"
        f"font-family:monospace;white-space:pre-wrap'>{body}</pre>"
    )


@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    return (
        f"<pre style='background:#111;color:#aaa;padding:24px;font-family:monospace'>"
        f"500 Internal Server Error\n\n{tb}</pre>"
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
