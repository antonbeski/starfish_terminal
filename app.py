import re
import time
import traceback
import requests
import random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

app = Flask(__name__)

# ── Live news channel YouTube handles ───────────────────────────────────────
NEWS_CHANNELS = [
    {"id": "cnbctv18",  "handle": "cnbctv18",  "label": "CNBC TV18",       "lang": "EN", "region": "India",  "video_id": "1_Ih0JYmkjI"},
    {"id": "bloomberg", "handle": "Bloomberg", "label": "Bloomberg Global", "lang": "EN", "region": "Global", "video_id": "iEpJwprxDdk"},
    {"id": "yahoofi",   "handle": "yahoofi",   "label": "Yahoo Finance",   "lang": "EN", "region": "Global", "video_id": "KQp-e_XQnDE"},
]

_YT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.youtube.com/",
}

def fetch_live_video_id(handle: str) -> tuple:
    for ch in NEWS_CHANNELS:
        if ch["handle"] == handle and ch.get("video_id"):
            return ch["video_id"], True

    def _get(url):
        return requests.get(url, headers=_YT_HEADERS, timeout=12, allow_redirects=True)

    video_id = None
    is_live = False
    try:
        r = _get(f"https://www.youtube.com/@{handle}/live")
        text = r.text
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", r.url)
        if not m:
            m = re.search(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', text)
        if m:
            candidate = m.group(1)
            actually_live = (
                '"isLive":true' in text
                or '"isLiveBroadcast"' in text
                or '"liveBroadcastContent":"live"' in text
            )
            if actually_live:
                video_id = candidate
                is_live = True
    except Exception:
        pass

    if not is_live:
        try:
            r2 = _get(f"https://www.youtube.com/@{handle}/videos")
            ids = re.findall(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', r2.text)
            seen, unique = set(), []
            for vid in ids:
                if vid not in seen:
                    seen.add(vid)
                    unique.append(vid)
            if unique:
                video_id = unique[0]
                is_live = False
        except Exception:
            pass

    return video_id, is_live


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

INDICATORS = [
    ("sma",  "SMA"),
    ("bb",   "Bollinger"),
    ("rsi",  "RSI"),
    ("macd", "MACD"),
    ("vol",  "Volume"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST YAHOO FINANCE SCRAPER
# Bypasses rate limits by scraping crumb + cookies directly from HTML,
# exactly like a real browser session. No dependency on yfinance internals.
# ══════════════════════════════════════════════════════════════════════════════

# Global session cache — reused across requests to keep cookies alive
_CACHE = {"session": None, "crumb": None, "ts": 0}
_CACHE_TTL = 1800  # re-authenticate every 30 min

_PERIOD_DAYS = {
    "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 366, "2y": 731, "5y": 1827,
}

# Rotate through multiple realistic User-Agent strings
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Yahoo Finance API base URLs (alternate between them to spread load)
_YF_BASES = [
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
]


def _new_session(ua: str | None = None) -> requests.Session:
    """Build a requests.Session that looks exactly like Chrome visiting Yahoo Finance."""
    s = requests.Session()
    ua = ua or random.choice(_UA_POOL)
    s.headers.update({
        "User-Agent":                ua,
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language":           "en-US,en;q=0.9",
        "Accept-Encoding":           "gzip, deflate, br",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-CH-UA":                 '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile":          "?0",
        "Sec-CH-UA-Platform":        '"Windows"',
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "none",
        "Sec-Fetch-User":            "?1",
        "Cache-Control":             "max-age=0",
        "DNT":                       "1",
    })
    return s


def _scrape_crumb(session: requests.Session, ticker: str) -> str | None:
    """
    Visit the Yahoo Finance quote page for the ticker and extract the crumb.
    Yahoo embeds the crumb as JSON inside the page HTML — we grab it with regex.
    This is the same thing a browser does automatically.
    """
    crumb = None

    # ① Try the fast crumb endpoint first (works if cookies are already set)
    for base in _YF_BASES:
        try:
            r = session.get(f"{base}/v1/test/getcrumb", timeout=8,
                            headers={"Referer": "https://finance.yahoo.com/"})
            if r.status_code == 200 and r.text and len(r.text) < 50 and "<" not in r.text:
                return r.text.strip()
        except Exception:
            pass

    # ② Visit the quote page to seed cookies, then extract crumb from HTML
    for url in [
        f"https://finance.yahoo.com/quote/{ticker}",
        "https://finance.yahoo.com/",
    ]:
        try:
            r = session.get(url, timeout=15, allow_redirects=True)
            html = r.text

            # Pattern 1: JSON key "crumb":"XYZ"
            m = re.search(r'"crumb"\s*:\s*"([^"]{5,30})"', html)
            if m:
                crumb = m.group(1).replace("\\u002F", "/")
                break

            # Pattern 2: CrumbStore:{crumb:"XYZ"}
            m = re.search(r'CrumbStore\s*:\s*\{\s*crumb\s*:\s*"([^"]{5,30})"', html)
            if m:
                crumb = m.group(1).replace("\\u002F", "/")
                break

            # Pattern 3: "crumb" inside a JS context
            m = re.search(r'["\']crumb["\']\s*,\s*["\']([^"\']{5,30})["\']', html)
            if m:
                crumb = m.group(1).replace("\\u002F", "/")
                break
        except Exception:
            continue

    # ③ Re-try the fast crumb endpoint now that we have cookies
    if not crumb:
        for base in _YF_BASES:
            try:
                r = session.get(
                    f"{base}/v1/test/getcrumb", timeout=8,
                    headers={"Referer": "https://finance.yahoo.com/"},
                )
                if r.status_code == 200 and r.text and len(r.text) < 50 and "<" not in r.text:
                    crumb = r.text.strip()
                    break
            except Exception:
                pass

    return crumb


def _get_authenticated_session(ticker: str, force_refresh: bool = False):
    """
    Return a (session, crumb) pair, reusing the cached one if still fresh.
    Automatically re-authenticates when the cache expires or on force_refresh.
    """
    now = time.time()
    if (
        not force_refresh
        and _CACHE["session"] is not None
        and _CACHE["crumb"]
        and (now - _CACHE["ts"]) < _CACHE_TTL
    ):
        return _CACHE["session"], _CACHE["crumb"]

    # Build a fresh session, warm it up, scrape the crumb
    session = _new_session()

    # Seed cookies via fc.yahoo.com (sets A1/A3 consent cookies)
    for seed_url in [
        "https://fc.yahoo.com",
        "https://finance.yahoo.com/",
    ]:
        try:
            session.get(seed_url, timeout=8, allow_redirects=True)
            break
        except Exception:
            pass

    crumb = _scrape_crumb(session, ticker)

    _CACHE["session"] = session
    _CACHE["crumb"]   = crumb
    _CACHE["ts"]      = now
    return session, crumb


def _parse_v8_response(j: dict) -> pd.DataFrame | None:
    """Parse a Yahoo Finance v8 chart JSON response into a DataFrame."""
    try:
        result = j.get("chart", {}).get("result")
        if not result:
            return None
        res        = result[0]
        timestamps = res.get("timestamp", [])
        if not timestamps:
            return None

        quote = res["indicators"]["quote"][0]
        adj   = res["indicators"].get("adjclose", [{}])
        close = (adj[0].get("adjclose") if adj else None) or quote.get("close")

        df = pd.DataFrame({
            "Open":   quote.get("open"),
            "High":   quote.get("high"),
            "Low":    quote.get("low"),
            "Close":  close,
            "Volume": quote.get("volume"),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True).normalize())
        df.index.name = "Date"
        df = df[df["Close"].notna()]
        return df if not df.empty else None
    except Exception:
        return None


def _fetch_v8(ticker: str, period: str,
              session: requests.Session, crumb: str | None) -> pd.DataFrame | None:
    """
    Fetch OHLCV data from Yahoo Finance v8/chart endpoint.
    This is the primary, most reliable path.
    """
    params = {
        "range":                period,
        "interval":             "1d",
        "includeAdjustedClose": "true",
        "events":               "div,splits",
    }
    if crumb:
        params["crumb"] = crumb

    api_headers = {
        "Referer":        "https://finance.yahoo.com/",
        "Accept":         "application/json, text/plain, */*",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    for base in _YF_BASES:
        url = f"{base}/v8/finance/chart/{ticker}"
        try:
            r = session.get(url, params=params, headers=api_headers, timeout=15)
            if r.status_code == 401:
                return None   # crumb is stale — caller will refresh
            if r.status_code != 200:
                continue
            df = _parse_v8_response(r.json())
            if df is not None:
                return df
        except Exception:
            continue
    return None


def _fetch_v7_csv(ticker: str, period: str,
                  session: requests.Session, crumb: str | None) -> pd.DataFrame | None:
    """
    Fallback: Yahoo Finance v7 download CSV endpoint.
    Works even without a valid crumb in many regions.
    """
    from io import StringIO
    end_ts   = int(time.time())
    start_ts = end_ts - _PERIOD_DAYS.get(period, 183) * 86400

    params = {
        "period1":              start_ts,
        "period2":              end_ts,
        "interval":             "1d",
        "events":               "history",
        "includeAdjustedClose": "true",
    }
    if crumb:
        params["crumb"] = crumb

    api_headers = {
        "Referer":        "https://finance.yahoo.com/",
        "Accept":         "text/csv,application/json,*/*",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }

    for base in _YF_BASES:
        url = f"{base}/v7/finance/download/{ticker}"
        try:
            r = session.get(url, params=params, headers=api_headers, timeout=15)
            if r.status_code != 200 or "Date" not in r.text:
                continue
            df = pd.read_csv(StringIO(r.text))
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
            # Use Adj Close when available, fall back to Close
            if "Adj Close" in df.columns:
                df["Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
            else:
                df["Close"] = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce")
            for col in ["Open", "High", "Low", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if not df.empty:
                return df
        except Exception:
            continue
    return None


def _fetch_yf_library(ticker: str, period: str,
                      session: requests.Session) -> pd.DataFrame | None:
    """
    Last-resort: use the yfinance library itself with our authenticated session.
    Suppresses all print output so rate-limit warnings don't surface to the user.
    """
    import io, contextlib
    buf = io.StringIO()
    df  = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            t  = yf.Ticker(ticker, session=session)
            df = t.history(period=period, interval="1d",
                           auto_adjust=True, actions=False, timeout=15)
        if df is not None and not df.empty:
            return _flatten(df)
    except Exception:
        pass

    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True,
                             actions=False, timeout=15, session=session)
        if df is not None and not df.empty:
            return _flatten(df)
    except Exception:
        pass
    return None


def fetch_yfinance_data(ticker: str, period: str):
    """
    Master fetch function — tries 4 strategies, auto-refreshes crumb on 401.
    Returns (DataFrame | None, error_str | None).

    Strategy order:
      1. v8 Chart API  (JSON, most reliable, supports crumb auth)
      2. v7 CSV API    (CSV download, good fallback)
      3. Re-auth + v8  (force a fresh crumb/session and retry v8)
      4. Re-auth + v7  (fresh session, v7 CSV fallback)
      5. yfinance lib  (last resort, uses our authenticated session)
    """
    last_err = None

    for attempt in range(2):          # attempt 0 = cached session, attempt 1 = fresh session
        force = (attempt == 1)
        try:
            session, crumb = _get_authenticated_session(ticker, force_refresh=force)
        except Exception as e:
            last_err = str(e)
            continue

        # ── Strategy A: v8 JSON ──────────────────────────────────────────
        try:
            df = _fetch_v8(ticker, period, session, crumb)
            if df is not None and not df.empty:
                return df, None
        except Exception as e:
            last_err = str(e)

        # ── Strategy B: v7 CSV ───────────────────────────────────────────
        try:
            df = _fetch_v7_csv(ticker, period, session, crumb)
            if df is not None and not df.empty:
                return df, None
        except Exception as e:
            last_err = str(e)

        # Force cache refresh on next iteration
        _CACHE["session"] = None
        _CACHE["crumb"]   = None
        time.sleep(0.4)

    # ── Strategy C: yfinance library (last resort) ───────────────────────
    try:
        session, _ = _get_authenticated_session(ticker, force_refresh=True)
        df = _fetch_yf_library(ticker, period, session)
        if df is not None and not df.empty:
            return df, None
    except Exception as e:
        last_err = str(e)

    sym_hint = " (use .NS for NSE, e.g. TCS.NS)" if "." not in ticker else ""
    return None, f"Could not fetch data for '{ticker}'{sym_hint}. {last_err or ''}"


def _flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _get_name(ticker):
    try:
        session, _ = _get_authenticated_session(ticker)
        t = yf.Ticker(ticker, session=session)
        name = (t.fast_info.get("longName") or "").strip()
        if not name:
            name = (t.info.get("shortName") or "").strip()
        return name or ticker
    except Exception:
        return ticker


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATOR CALCULATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_sma(close, window):
    return close.rolling(window=window).mean()


def calc_ema(close, window):
    return close.ewm(span=window, adjust=False).mean()


def calc_bollinger(close, window=20, num_std=2):
    sma   = calc_sma(close, window)
    std   = close.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def calc_rsi(close, window=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════

_C = {
    "bg":        "rgba(0,0,0,0)",
    "paper":     "rgba(0,0,0,0)",
    "grid":      "rgba(255,255,255,0.06)",
    "axis":      "#555555",
    "text":      "#888888",
    "white":     "#ffffff",
    "green":     "#26a69a",
    "red":       "#ef5350",
    "sma20":     "#FFD700",
    "sma50":     "#FF8C00",
    "sma200":    "#00BFFF",
    "bb_upper":  "rgba(120,180,255,0.7)",
    "bb_lower":  "rgba(120,180,255,0.7)",
    "bb_fill":   "rgba(120,180,255,0.06)",
    "rsi":       "#a78bfa",
    "rsi_ob":    "rgba(239,83,80,0.25)",
    "rsi_os":    "rgba(38,166,154,0.25)",
    "macd":      "#60a5fa",
    "signal":    "#f97316",
    "hist_pos":  "rgba(38,166,154,0.8)",
    "hist_neg":  "rgba(239,83,80,0.8)",
    "vol_up":    "rgba(38,166,154,0.5)",
    "vol_dn":    "rgba(239,83,80,0.5)",
}

_AXIS_STYLE = dict(
    gridcolor=_C["grid"], color=_C["axis"],
    showline=False, zeroline=False,
    tickfont=dict(size=10, color=_C["text"]),
)


def build_chart(ticker, period, chart_type, indicators):
    data, err = fetch_yfinance_data(ticker, period)
    if err:
        return None, f"Data error: {err}"
    if data is None or data.empty:
        return None, (
            f"No data found for '{ticker}'. "
            "Use '.NS' suffix for NSE stocks (e.g. TCS.NS), '.BO' for BSE."
        )

    missing = {"Open", "High", "Low", "Close"} - set(data.columns)
    if missing:
        return None, f"Missing columns: {missing}"

    data = data.dropna(subset=["Close"])
    if len(data) < 5:
        return None, "Not enough data points to render a chart."

    close  = data["Close"].squeeze()
    high   = data["High"].squeeze()
    low    = data["Low"].squeeze()
    open_  = data["Open"].squeeze()
    vol    = data["Volume"].squeeze() if "Volume" in data.columns else None
    dates  = data.index

    name     = _get_name(ticker)
    currency = "INR" if ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO") else "USD"

    # ── Decide subplot layout ──────────────────────────────────────────────
    show_vol  = "vol"  in indicators and vol is not None
    show_rsi  = "rsi"  in indicators
    show_macd = "macd" in indicators

    rows      = 1 + int(show_vol) + int(show_rsi) + int(show_macd)
    row_heights_map = {1: [1.0], 2: [0.65, 0.35], 3: [0.55, 0.22, 0.23], 4: [0.50, 0.17, 0.17, 0.16]}
    row_heights = row_heights_map.get(rows, [0.50, 0.17, 0.17, 0.16])

    specs = [[{"secondary_y": False}] for _ in range(rows)]

    subplot_titles = [f"{name} ({ticker.upper()})"]
    if show_vol:  subplot_titles.append("Volume")
    if show_rsi:  subplot_titles.append("RSI (14)")
    if show_macd: subplot_titles.append("MACD (12, 26, 9)")

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    r_vol  = 1 + int(show_vol and True)
    r_rsi  = 1 + int(show_vol) + int(show_rsi and True)
    r_macd = 1 + int(show_vol) + int(show_rsi) + int(show_macd and True)
    r_vol  = 2 if show_vol  else None
    r_rsi  = (2 + int(show_vol)) if show_rsi  else None
    r_macd = (2 + int(show_vol) + int(show_rsi)) if show_macd else None

    # ── Main price trace ───────────────────────────────────────────────────
    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(
            x=dates, open=open_, high=high, low=low, close=close,
            name="Price",
            increasing_line_color=_C["green"],  increasing_fillcolor=f'rgba(38,166,154,0.18)',
            decreasing_line_color=_C["red"],    decreasing_fillcolor=f'rgba(239,83,80,0.18)',
            line=dict(width=1),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=dates, y=close, mode="lines", name="Price",
            line=dict(color=_C["white"], width=2),
            fill="tozeroy", fillcolor="rgba(255,255,255,0.04)",
        ), row=1, col=1)

    # ── SMA overlays ──────────────────────────────────────────────────────
    if "sma" in indicators:
        for w, color, label in [(20, _C["sma20"], "SMA 20"), (50, _C["sma50"], "SMA 50"), (200, _C["sma200"], "SMA 200")]:
            if len(close) >= w:
                fig.add_trace(go.Scatter(
                    x=dates, y=calc_sma(close, w),
                    mode="lines", name=label,
                    line=dict(color=color, width=1.2, dash="solid"),
                    opacity=0.85,
                ), row=1, col=1)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    if "bb" in indicators and len(close) >= 20:
        bb_u, bb_m, bb_l = calc_bollinger(close)
        fig.add_trace(go.Scatter(
            x=dates, y=bb_u, mode="lines", name="BB Upper",
            line=dict(color=_C["bb_upper"], width=1, dash="dot"), showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=bb_l, mode="lines", name="BB Lower",
            line=dict(color=_C["bb_lower"], width=1, dash="dot"),
            fill="tonexty", fillcolor=_C["bb_fill"], showlegend=True,
        ), row=1, col=1)

    # ── Volume ─────────────────────────────────────────────────────────────
    if show_vol and vol is not None:
        colors = [_C["vol_up"] if c >= o else _C["vol_dn"]
                  for c, o in zip(close, open_)]
        fig.add_trace(go.Bar(
            x=dates, y=vol, name="Volume",
            marker_color=colors, showlegend=False,
        ), row=r_vol, col=1)

    # ── RSI ────────────────────────────────────────────────────────────────
    if show_rsi and len(close) >= 15:
        rsi_vals = calc_rsi(close)
        fig.add_trace(go.Scatter(
            x=dates, y=rsi_vals, mode="lines", name="RSI",
            line=dict(color=_C["rsi"], width=1.5), showlegend=False,
        ), row=r_rsi, col=1)
        # Overbought / oversold bands
        fig.add_hrect(y0=70, y1=100, row=r_rsi, col=1,
                      fillcolor=_C["rsi_ob"], line_width=0, layer="below")
        fig.add_hrect(y0=0, y1=30, row=r_rsi, col=1,
                      fillcolor=_C["rsi_os"], line_width=0, layer="below")
        for lvl, color in [(70, "rgba(239,83,80,0.5)"), (30, "rgba(38,166,154,0.5)"), (50, "rgba(255,255,255,0.15)")]:
            fig.add_hline(y=lvl, row=r_rsi, col=1,
                          line=dict(color=color, width=0.8, dash="dash"))

    # ── MACD ───────────────────────────────────────────────────────────────
    if show_macd and len(close) >= 27:
        macd_line, sig_line, hist = calc_macd(close)
        hist_colors = [_C["hist_pos"] if v >= 0 else _C["hist_neg"] for v in hist.fillna(0)]
        fig.add_trace(go.Bar(
            x=dates, y=hist, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=r_macd, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=macd_line, mode="lines", name="MACD",
            line=dict(color=_C["macd"], width=1.5), showlegend=False,
        ), row=r_macd, col=1)
        fig.add_trace(go.Scatter(
            x=dates, y=sig_line, mode="lines", name="Signal",
            line=dict(color=_C["signal"], width=1.5), showlegend=False,
        ), row=r_macd, col=1)
        fig.add_hline(y=0, row=r_macd, col=1,
                      line=dict(color="rgba(255,255,255,0.2)", width=0.8, dash="dash"))

    # ── Layout ─────────────────────────────────────────────────────────────
    total_height = 420 + 120 * (rows - 1)
    fig.update_layout(
        height=total_height,
        plot_bgcolor=_C["bg"],
        paper_bgcolor=_C["paper"],
        font=dict(color=_C["text"], family="'DM Sans', sans-serif", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=10, color=_C["text"]),
        ),
        hovermode="x unified",
        margin=dict(l=55, r=20, t=55, b=30),
        hoverlabel=dict(
            bgcolor="rgba(12,12,12,0.95)",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(color="#ffffff"),
        ),
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )

    # Update all x/y axes
    axis_common = dict(gridcolor=_C["grid"], color=_C["axis"], showline=False,
                       zeroline=False, tickfont=dict(size=9, color=_C["text"]))
    for i in range(1, rows + 1):
        xkey = f"xaxis{'' if i==1 else i}"
        ykey = f"yaxis{'' if i==1 else i}"
        fig.update_layout(**{xkey: {**axis_common, "rangeslider": {"visible": False}}})
        fig.update_layout(**{ykey: {**axis_common}})

    if show_rsi:
        rsi_ykey = f"yaxis{'' if r_rsi==1 else r_rsi}"
        fig.update_layout(**{rsi_ykey: {**axis_common, "range": [0, 100]}})

    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font.color = "#555555"
        ann.font.size  = 10

    return pyo.plot(fig, output_type="div", include_plotlyjs=False), None


# ══════════════════════════════════════════════════════════════════════════════
# HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_page(ticker, period, chart_type, active_indicators, graph_html, error):
    chips = ""
    for sym, _ in POPULAR_STOCKS:
        cls = "chip active" if sym == ticker else "chip"
        chips += f'<span class="{cls}" onclick="setTicker(\'{sym}\')">{sym}</span>\n'

    period_opts = ""
    for val, label in PERIODS:
        sel = "selected" if val == period else ""
        period_opts += f'<option value="{val}" {sel}>{label}</option>\n'

    ct_candle = "selected" if chart_type == "candlestick" else ""
    ct_line   = "selected" if chart_type == "line"        else ""

    ind_chips = ""
    for key, label in INDICATORS:
        cls = "ind-chip active" if key in active_indicators else "ind-chip"
        ind_chips += f'<span class="{cls}" data-ind="{key}" onclick="toggleInd(this)">{label}</span>\n'

    if error:
        content = f'<div class="error-box">{error}</div>'
    elif graph_html:
        content = graph_html
    else:
        content = '<div class="empty-state">Enter a ticker above to load a chart.</div>'

    tabs_html = ""
    for i, ch in enumerate(NEWS_CHANNELS):
        cls = "news-tab active" if i == 0 else "news-tab"
        tags = f'<span class="news-tag">{ch["region"]}</span><span class="news-tag">{ch["lang"]}</span>'
        tabs_html += f'<button class="{cls}" data-handle="{ch["handle"]}">{ch["label"]} {tags}</button>\n'

    first_handle = NEWS_CHANNELS[0]["handle"]
    active_ind_js = str(list(active_indicators))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>STARFISH</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #060606; --surface: rgba(255,255,255,0.04); --border: rgba(255,255,255,0.09);
      --border-soft: rgba(255,255,255,0.05); --text: #f0f0f0; --text-muted: #666666;
      --text-dim: #3a3a3a; --accent: #ffffff; --accent-mute: rgba(255,255,255,0.1);
      --blur: blur(20px); --radius: 16px; --radius-sm: 9px;
    }}
    body {{ font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text);
            min-height: 100vh; -webkit-font-smoothing: antialiased; overflow-x: hidden; }}
    body::before {{ content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background: radial-gradient(ellipse 90% 55% at 15% 5%, rgba(255,255,255,0.022) 0%, transparent 55%),
                  radial-gradient(ellipse 55% 45% at 85% 85%, rgba(255,255,255,0.012) 0%, transparent 50%); }}
    header {{ position: sticky; top: 0; z-index: 100; height: 58px; display: flex;
              align-items: center; justify-content: space-between; padding: 0 28px;
              background: rgba(6,6,6,0.75); backdrop-filter: var(--blur);
              border-bottom: 1px solid var(--border-soft); }}
    .logo {{ display: flex; align-items: center; gap: 10px; font-size: 0.9rem; font-weight: 600;
             letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent); }}
    .logo-pip {{ width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
                 animation: blink 2.8s ease-in-out infinite; flex-shrink: 0; }}
    @keyframes blink {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:0.2; transform:scale(0.65); }} }}
    .subtitle {{ font-size: 0.72rem; color: var(--text-dim); letter-spacing: 0.03em; }}
    main {{ position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 30px 20px 64px; }}
    .glass {{ background: var(--surface); backdrop-filter: var(--blur); border: 1px solid var(--border); border-radius: var(--radius); }}
    .panel {{ padding: 26px 30px; margin-bottom: 18px; }}
    .panel-label {{ font-size: 0.62rem; font-weight: 600; letter-spacing: 0.16em;
                    text-transform: uppercase; color: var(--text-dim); margin-bottom: 20px; }}
    form {{ display: grid; grid-template-columns: 1.5fr 1fr 1fr auto; gap: 14px; align-items: end; }}
    .field-group label {{ display: block; font-size: 0.7rem; font-weight: 500;
                          letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px; }}
    input, select {{ width: 100%; background: rgba(255,255,255,0.035); border: 1px solid var(--border);
                     border-radius: var(--radius-sm); color: var(--text); padding: 10px 14px;
                     font-size: 0.875rem; font-family: inherit; outline: none;
                     transition: border-color .2s, background .2s, box-shadow .2s;
                     appearance: none; -webkit-appearance: none; }}
    input::placeholder {{ color: var(--text-dim); }}
    input:focus, select:focus {{ border-color: rgba(255,255,255,0.28); background: rgba(255,255,255,0.065);
                                  box-shadow: 0 0 0 3px rgba(255,255,255,0.05); }}
    select {{ cursor: pointer;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 13px center; padding-right: 34px; }}
    select option {{ background: #111111; color: #f0f0f0; }}
    .btn {{ background: var(--accent); color: #000; border: none; border-radius: var(--radius-sm);
            padding: 10px 26px; font-size: 0.8rem; font-weight: 600; font-family: inherit;
            cursor: pointer; white-space: nowrap; letter-spacing: 0.09em;
            text-transform: uppercase; transition: opacity .18s, transform .13s; height: 42px; }}
    .btn:hover {{ opacity: 0.85; }} .btn:active {{ transform: scale(0.96); }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 22px;
              padding-top: 20px; border-top: 1px solid var(--border-soft); }}
    .chip {{ background: transparent; border: 1px solid var(--border); border-radius: 100px;
             padding: 5px 15px; font-size: 0.72rem; font-family: 'DM Mono', monospace;
             cursor: pointer; color: var(--text-muted); letter-spacing: 0.05em;
             transition: all .16s; user-select: none; }}
    .chip:hover {{ border-color: rgba(255,255,255,0.3); color: var(--text); background: var(--accent-mute); }}
    .chip.active {{ background: var(--accent); border-color: var(--accent); color: #000; font-weight: 600; }}
    /* Indicators row */
    .ind-row {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 16px;
                padding-top: 16px; border-top: 1px solid var(--border-soft);
                align-items: center; }}
    .ind-label {{ font-size: 0.62rem; font-weight: 600; letter-spacing: 0.12em;
                  text-transform: uppercase; color: var(--text-dim); margin-right: 4px; }}
    .ind-chip {{ background: transparent; border: 1px solid var(--border); border-radius: 100px;
                 padding: 4px 14px; font-size: 0.7rem; font-family: 'DM Mono', monospace;
                 cursor: pointer; color: var(--text-muted); letter-spacing: 0.05em;
                 transition: all .16s; user-select: none; }}
    .ind-chip:hover {{ border-color: rgba(255,255,255,0.3); color: var(--text); background: var(--accent-mute); }}
    .ind-chip.active {{ background: rgba(255,255,255,0.12); border-color: rgba(255,255,255,0.35);
                        color: var(--text); font-weight: 600; }}
    /* Chart */
    .chart-card {{ padding: 20px 16px 12px; min-height: 460px;
                   display: flex; align-items: flex-start; justify-content: center; overflow: hidden; }}
    .chart-card > div {{ width: 100%; }}
    .error-box {{ border: 1px solid rgba(255,255,255,0.1); border-left: 3px solid rgba(255,255,255,0.45);
                  border-radius: var(--radius-sm); padding: 16px 20px; color: #999;
                  font-size: 0.875rem; background: rgba(255,255,255,0.025); width: 100%; line-height: 1.6; }}
    .empty-state {{ color: var(--text-dim); font-size: 0.85rem; text-align: center; letter-spacing: 0.03em; }}
    /* News */
    .news-panel {{ padding: 26px 30px; margin-top: 18px; }}
    .news-live-dot {{ display: inline-block; width: 6px; height: 6px; border-radius: 50%;
                      background: #ff4444; margin-right: 6px;
                      animation: livepulse 1.4s ease-in-out infinite; vertical-align: middle; }}
    @keyframes livepulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:0.3; transform:scale(0.6); }} }}
    .news-tabs {{ display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }}
    .news-tab {{ background: transparent; border: 1px solid var(--border); border-radius: 100px;
                 padding: 6px 18px; font-size: 0.72rem; font-family: 'DM Mono', monospace;
                 cursor: pointer; color: var(--text-muted); letter-spacing: 0.05em;
                 transition: all .16s; user-select: none; }}
    .news-tab:hover {{ border-color: rgba(255,255,255,0.3); color: var(--text); background: var(--accent-mute); }}
    .news-tab.active {{ background: var(--accent); border-color: var(--accent); color: #000; font-weight: 600; }}
    .news-iframe-wrap {{ position: relative; width: 100%; padding-top: 56.25%;
                         border-radius: var(--radius-sm); overflow: hidden; background: rgba(0,0,0,0.5); }}
    .news-loading {{ position: absolute; inset: 0; display: flex; align-items: center;
                     justify-content: center; color: var(--text-muted); font-size: 0.8rem;
                     letter-spacing: 0.05em; flex-direction: column; gap: 12px; }}
    .news-spinner {{ width: 22px; height: 22px; border-radius: 50%;
                     border: 2px solid rgba(255,255,255,0.1);
                     border-top-color: rgba(255,255,255,0.5);
                     animation: spin 0.8s linear infinite; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    .news-iframe-wrap iframe {{ position: absolute; inset: 0; width: 100%; height: 100%; border: none; }}
    .news-tag {{ font-size: 0.55rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
                 padding: 1px 5px; border-radius: 4px;
                 background: rgba(255,255,255,0.08); color: var(--text-dim);
                 margin-left: 4px; vertical-align: middle; }}
    .news-tab.active .news-tag {{ background: rgba(0,0,0,0.15); color: rgba(0,0,0,0.5); }}
    .news-status-badge {{ display: inline-flex; align-items: center; gap: 5px; font-size: 0.62rem;
                          font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
                          padding: 3px 10px; border-radius: 100px; white-space: nowrap; }}
    .news-status-badge.live {{ background: rgba(255,60,60,0.15); border: 1px solid rgba(255,60,60,0.35); color: #ff6b6b; }}
    .news-status-badge.live::before {{ content: ''; display: inline-block; width: 5px; height: 5px;
                                       border-radius: 50%; background: #ff4444;
                                       animation: livepulse 1.4s ease-in-out infinite; }}
    .news-status-badge.latest {{ background: rgba(255,255,255,0.06); border: 1px solid var(--border); color: var(--text-muted); }}
    .news-status-badge.error {{ background: rgba(255,160,0,0.1); border: 1px solid rgba(255,160,0,0.25); color: #ffaa44; }}
    @media(max-width:860px){{
      form{{grid-template-columns:1fr 1fr;gap:12px;}}
      .field-group:first-child{{grid-column:span 2;}}
      .btn{{grid-column:span 2;width:100%;}}
    }}
    @media(max-width:600px){{
      header{{padding:0 16px;}} .subtitle{{display:none;}}
      main{{padding:18px 14px 48px;}} .panel{{padding:20px 18px;}}
      .chart-card{{padding:16px 10px 10px;min-height:300px;}}
      .news-panel{{padding:20px 18px;}}
    }}
    @media(max-width:400px){{
      form{{grid-template-columns:1fr;}}
      .field-group:first-child{{grid-column:span 1;}}
      .btn{{grid-column:span 1;}}
    }}
  </style>
</head>
<body>
<header>
  <div class="logo"><span class="logo-pip"></span>Starfish</div>
  <span class="subtitle">LIVE STOCK MARKETS</span>
</header>
<main>
  <div class="glass panel">
    <div class="panel-label">Search</div>
    <form method="POST" action="/" id="main-form">
      <input type="hidden" name="indicators" id="indicators-hidden" value="{','.join(active_indicators)}"/>
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
      <button type="submit" class="btn">Load</button>
    </form>
    <div class="chips">{chips}</div>
    <div class="ind-row">
      <span class="ind-label">Indicators</span>
      {ind_chips}
    </div>
  </div>

  <div class="glass chart-card" id="chart-wrap">{content}</div>

  <div class="glass news-panel">
    <div class="panel-label"><span class="news-live-dot"></span>Live Financial News</div>
    <div class="news-tabs" id="news-tabs">{tabs_html}</div>
    <div class="news-iframe-wrap">
      <div id="news-loading" class="news-loading">
        <div class="news-spinner"></div><span>Loading stream&hellip;</span>
      </div>
      <iframe id="news-iframe"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen style="display:none"></iframe>
    </div>
    <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <span id="news-status-badge" class="news-status-badge" style="display:none"></span>
    </div>
  </div>
</main>

<script>
  // ── Ticker chips ──────────────────────────────────────────────────────────
  function setTicker(s) {{
    document.getElementById('ticker').value = s;
    document.getElementById('main-form').submit();
  }}

  // ── Indicator chips ───────────────────────────────────────────────────────
  var activeInds = {active_ind_js};

  function toggleInd(el) {{
    var key = el.dataset.ind;
    var idx = activeInds.indexOf(key);
    if (idx === -1) {{ activeInds.push(key); el.classList.add('active'); }}
    else            {{ activeInds.splice(idx,1); el.classList.remove('active'); }}
    document.getElementById('indicators-hidden').value = activeInds.join(',');
    document.getElementById('main-form').submit();
  }}

  // ── Live news player ──────────────────────────────────────────────────────
  var iframe  = document.getElementById('news-iframe');
  var loading = document.getElementById('news-loading');
  var badge   = document.getElementById('news-status-badge');
  var currentHandle = null;

  function setLoading(msg) {{
    iframe.style.display = 'none';
    loading.innerHTML = '<div class="news-spinner"></div><span>' + msg + '</span>';
    loading.style.display = 'flex';
    badge.style.display = 'none';
  }}

  function setError(msg) {{
    iframe.style.display = 'none';
    loading.innerHTML = '<span>' + msg + '</span>';
    loading.style.display = 'flex';
    badge.className = 'news-status-badge error';
    badge.textContent = '\u26a0 Unavailable';
    badge.style.display = 'inline-flex';
  }}

  function loadChannel(handle) {{
    if (currentHandle === handle) return;
    currentHandle = handle;
    setLoading('Loading stream\u2026');
    iframe.src = 'about:blank';
    fetch('/api/live-id?handle=' + encodeURIComponent(handle))
      .then(function(r) {{ if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); }})
      .then(function(data) {{
        if (handle !== currentHandle) return;
        if (data.error || !data.video_id) {{ setError('Stream unavailable.'); return; }}
        iframe.src = 'https://www.youtube.com/embed/' + data.video_id + '?autoplay=1&rel=0&modestbranding=1';
        iframe.style.display = 'block';
        loading.style.display = 'none';
        badge.style.display = 'inline-flex';
        if (data.is_live) {{
          badge.className = 'news-status-badge live'; badge.textContent = 'LIVE';
        }} else {{
          badge.className = 'news-status-badge latest'; badge.textContent = '\u25b6 Latest Video';
        }}
      }})
      .catch(function() {{ if (handle !== currentHandle) return; setError('Could not load stream.'); }});
  }}

  document.getElementById('news-tabs').addEventListener('click', function(e) {{
    var btn = e.target.closest('.news-tab');
    if (!btn) return;
    document.querySelectorAll('.news-tab').forEach(function(t) {{ t.classList.remove('active'); }});
    btn.classList.add('active');
    currentHandle = null;
    loadChannel(btn.dataset.handle);
  }});

  loadChannel('{first_handle}');
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_INDICATORS = {"sma", "vol"}


@app.route("/", methods=["GET", "POST"])
def index():
    ticker     = (request.form.get("ticker", "AAPL") or "AAPL").strip().upper()
    period     = request.form.get("period", "6mo")
    chart_type = request.form.get("chart_type", "candlestick")
    ind_raw    = request.form.get("indicators", ",".join(DEFAULT_INDICATORS))

    if period not in VALID_PERIODS:
        period = "6mo"
    if chart_type not in ("candlestick", "line"):
        chart_type = "candlestick"

    active_indicators = set(filter(None, ind_raw.split(","))) if ind_raw else DEFAULT_INDICATORS

    graph_html, error = build_chart(ticker, period, chart_type, active_indicators)
    return render_page(ticker, period, chart_type, active_indicators, graph_html, error)


@app.route("/api/live-id")
def api_live_id():
    handle = request.args.get("handle", "").strip()
    if not handle:
        return jsonify({"error": "missing handle"}), 400
    video_id, is_live = fetch_live_video_id(handle)
    if video_id:
        return jsonify({"video_id": video_id, "is_live": is_live})
    return jsonify({"error": "not found"}), 404


@app.route("/debug")
def debug():
    out = []
    color = "#7fff7f"
    try:
        df, err = fetch_yfinance_data("AAPL", "5d")
        if err:
            out.append(f"Error: {err}")
            color = "#ff7f7f"
        elif df is not None:
            out.append(f"Strategy OK — shape: {df.shape}")
            out.append(df.tail().to_string())
        else:
            out.append("No data returned")
            color = "#ffaa44"
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
