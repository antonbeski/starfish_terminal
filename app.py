import os
import re
import time
import traceback
import requests
import random
import json
import threading
import numpy as np
import pandas as pd
from collections import deque
from flask import Flask, request, jsonify
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

app = Flask(__name__)

# ── OpenRouter AI config ─────────────────────────────────────────────────────
OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY", "")

AI_MODELS = [
    {"id": "deepseek/deepseek-r1",                  "key": "deepseek", "label": "DeepSeek R1",   "desc": "Chain-of-thought reasoning", "color": "#7c3aed"},
    {"id": "meta-llama/llama-3.3-70b-instruct",     "key": "llama",    "label": "Llama 3.3 70B", "desc": "Fast & balanced",            "color": "#0ea5e9"},
    {"id": "qwen/qwen3-coder",                       "key": "qwen",     "label": "Qwen3 Coder",   "desc": "Quantitative focus",         "color": "#f59e0b"},
]
RL_RPM = 20
RL_RPD = 200

_rl_lock  = threading.Lock()
_rl_state = {m["key"]: {"rpm": deque(), "rpd": deque()} for m in AI_MODELS}


def _rl_clean(key):
    now = time.time()
    while _rl_state[key]["rpm"] and now - _rl_state[key]["rpm"][0] > 60:
        _rl_state[key]["rpm"].popleft()
    while _rl_state[key]["rpd"] and now - _rl_state[key]["rpd"][0] > 86400:
        _rl_state[key]["rpd"].popleft()


def rl_check(key):
    with _rl_lock:
        _rl_clean(key)
        ru = len(_rl_state[key]["rpm"])
        du = len(_rl_state[key]["rpd"])
    return {"rpm_used": ru, "rpm_max": RL_RPM, "rpd_used": du, "rpd_max": RL_RPD,
            "available": ru < RL_RPM and du < RL_RPD}


def rl_record(key):
    with _rl_lock:
        t = time.time()
        _rl_state[key]["rpm"].append(t)
        _rl_state[key]["rpd"].append(t)


def rl_next_rpm_reset(key):
    with _rl_lock:
        if not _rl_state[key]["rpm"]:
            return 0
        return max(0, int(60 - (time.time() - _rl_state[key]["rpm"][0])))


# ── YouTube live news ────────────────────────────────────────────────────────
NEWS_CHANNELS = [
    {"id": "cnbctv18",  "handle": "cnbctv18",  "label": "CNBC TV18",       "lang": "EN", "region": "India",  "video_id": "1_Ih0JYmkjI"},
    {"id": "bloomberg", "handle": "Bloomberg", "label": "Bloomberg Global", "lang": "EN", "region": "Global", "video_id": "iEpJwprxDdk"},
    {"id": "yahoofi",   "handle": "yahoofi",   "label": "Yahoo Finance",   "lang": "EN", "region": "Global", "video_id": "KQp-e_XQnDE"},
]
_YT_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.youtube.com/"}


def fetch_live_video_id(handle):
    for ch in NEWS_CHANNELS:
        if ch["handle"] == handle and ch.get("video_id"):
            return ch["video_id"], True
    def _get(u): return requests.get(u, headers=_YT_HDR, timeout=12, allow_redirects=True)
    vid, live = None, False
    try:
        r = _get(f"https://www.youtube.com/@{handle}/live"); text = r.text
        m = re.search(r'[?&]v=([A-Za-z0-9_-]{11})', r.url) or re.search(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', text)
        if m and ('"isLive":true' in text or '"liveBroadcastContent":"live"' in text):
            vid, live = m.group(1), True
    except Exception: pass
    if not live:
        try:
            r2 = _get(f"https://www.youtube.com/@{handle}/videos")
            ids = list(dict.fromkeys(re.findall(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', r2.text)))
            if ids: vid, live = ids[0], False
        except Exception: pass
    return vid, live


POPULAR_STOCKS = [("AAPL","Apple"),("GOOGL","Google"),("MSFT","Microsoft"),("TSLA","Tesla"),
                  ("AMZN","Amazon"),("NVDA","NVIDIA"),("TCS.NS","TCS"),("RELIANCE.NS","Reliance")]
PERIODS = [("1mo","1 Month"),("3mo","3 Months"),("6mo","6 Months"),("1y","1 Year"),("2y","2 Years"),("5y","5 Years")]
VALID_PERIODS = {p[0] for p in PERIODS}
INDICATORS = [("sma","SMA"),("bb","Bollinger"),("rsi","RSI"),("macd","MACD"),("vol","Volume")]


# ══════════════════════════════════════════════════════════════════════════════
# YAHOO FINANCE SCRAPER
# ══════════════════════════════════════════════════════════════════════════════
_CACHE = {"session": None, "crumb": None, "ts": 0}
_CACHE_TTL = 1800
_PERIOD_DAYS = {"1mo":31,"3mo":92,"6mo":183,"1y":366,"2y":731,"5y":1827}
_YF_BASES = ["https://query1.finance.yahoo.com","https://query2.finance.yahoo.com"]
_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.207 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def _new_session(ua=None):
    s = requests.Session()
    s.headers.update({"User-Agent": ua or random.choice(_UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9", "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive", "Upgrade-Insecure-Requests": "1",
        "Sec-CH-UA": '"Chromium";v="124","Google Chrome";v="124","Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0", "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none", "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0", "DNT": "1"})
    return s


def _scrape_crumb(session, ticker):
    crumb = None
    for base in _YF_BASES:
        try:
            r = session.get(f"{base}/v1/test/getcrumb", timeout=8, headers={"Referer": "https://finance.yahoo.com/"})
            if r.status_code == 200 and r.text and len(r.text) < 50 and "<" not in r.text:
                return r.text.strip()
        except Exception: pass
    for url in [f"https://finance.yahoo.com/quote/{ticker}", "https://finance.yahoo.com/"]:
        try:
            html = session.get(url, timeout=15, allow_redirects=True).text
            for pat in [r'"crumb"\s*:\s*"([^"]{5,30})"',
                        r'CrumbStore\s*:\s*\{\s*crumb\s*:\s*"([^"]{5,30})"']:
                m = re.search(pat, html)
                if m: crumb = m.group(1).replace("\\u002F", "/"); break
            if crumb: break
        except Exception: continue
    if not crumb:
        for base in _YF_BASES:
            try:
                r = session.get(f"{base}/v1/test/getcrumb", timeout=8, headers={"Referer": "https://finance.yahoo.com/"})
                if r.status_code == 200 and r.text and len(r.text) < 50 and "<" not in r.text:
                    crumb = r.text.strip(); break
            except Exception: pass
    return crumb


def _get_auth(ticker, force=False):
    now = time.time()
    if not force and _CACHE["session"] and _CACHE["crumb"] and (now - _CACHE["ts"]) < _CACHE_TTL:
        return _CACHE["session"], _CACHE["crumb"]
    s = _new_session()
    for u in ["https://fc.yahoo.com", "https://finance.yahoo.com/"]:
        try: s.get(u, timeout=8, allow_redirects=True); break
        except Exception: pass
    c = _scrape_crumb(s, ticker)
    _CACHE.update({"session": s, "crumb": c, "ts": now})
    return s, c


def _parse_v8(j):
    try:
        res = j.get("chart", {}).get("result", [None])[0]
        if not res: return None
        ts = res.get("timestamp", [])
        if not ts: return None
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])
        cl = (adj[0].get("adjclose") if adj else None) or q.get("close")
        df = pd.DataFrame({"Open": q.get("open"), "High": q.get("high"),
                           "Low": q.get("low"), "Close": cl, "Volume": q.get("volume")},
                          index=pd.to_datetime(ts, unit="s", utc=True).normalize())
        df.index.name = "Date"
        df = df[df["Close"].notna()]
        return df if not df.empty else None
    except Exception: return None


def _fetch_v8(ticker, period, session, crumb):
    p = {"range": period, "interval": "1d", "includeAdjustedClose": "true", "events": "div,splits"}
    if crumb: p["crumb"] = crumb
    h = {"Referer": "https://finance.yahoo.com/", "Accept": "application/json,*/*",
         "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-site"}
    for base in _YF_BASES:
        try:
            r = session.get(f"{base}/v8/finance/chart/{ticker}", params=p, headers=h, timeout=15)
            if r.status_code == 401: return None
            if r.status_code == 200:
                df = _parse_v8(r.json())
                if df is not None: return df
        except Exception: continue
    return None


def _fetch_v7(ticker, period, session, crumb):
    from io import StringIO
    e, s2 = int(time.time()), int(time.time()) - _PERIOD_DAYS.get(period, 183) * 86400
    p = {"period1": s2, "period2": e, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
    if crumb: p["crumb"] = crumb
    h = {"Referer": "https://finance.yahoo.com/", "Accept": "text/csv,*/*",
         "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Site": "same-site"}
    for base in _YF_BASES:
        try:
            r = session.get(f"{base}/v7/finance/download/{ticker}", params=p, headers=h, timeout=15)
            if r.status_code != 200 or "Date" not in r.text: continue
            df = pd.read_csv(StringIO(r.text))
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
            df["Close"] = pd.to_numeric(df.get("Adj Close", df.get("Close", pd.Series())), errors="coerce")
            for col in ["Open","High","Low","Volume"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
            if not df.empty: return df
        except Exception: continue
    return None


def _fetch_lib(ticker, period, session):
    import io, contextlib
    buf = io.StringIO()
    for fn in [
        lambda: _flat(yf.Ticker(ticker, session=session).history(period=period, interval="1d", auto_adjust=True, actions=False, timeout=15)),
        lambda: _flat(yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, actions=False, timeout=15, session=session)),
    ]:
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                df = fn()
            if df is not None and not df.empty: return df
        except Exception: pass
    return None


def fetch_yfinance_data(ticker, period):
    last_err = None
    for attempt in range(2):
        try:
            session, crumb = _get_auth(ticker, force=(attempt == 1))
        except Exception as e: last_err = str(e); continue
        for fn in [lambda: _fetch_v8(ticker, period, session, crumb),
                   lambda: _fetch_v7(ticker, period, session, crumb)]:
            try:
                df = fn()
                if df is not None and not df.empty: return df, None
            except Exception as e: last_err = str(e)
        _CACHE.update({"session": None, "crumb": None}); time.sleep(0.4)
    try:
        session, _ = _get_auth(ticker, force=True)
        df = _fetch_lib(ticker, period, session)
        if df is not None and not df.empty: return df, None
    except Exception as e: last_err = str(e)
    hint = " (use .NS for NSE, e.g. TCS.NS)" if "." not in ticker else ""
    return None, f"Could not fetch '{ticker}'{hint}. {last_err or ''}"


def _flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _get_name(ticker):
    try:
        s, _ = _get_auth(ticker)
        t = yf.Ticker(ticker, session=s)
        return (t.fast_info.get("longName") or t.info.get("shortName") or "").strip() or ticker
    except Exception: return ticker


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def calc_sma(c, w):  return c.rolling(w).mean()
def calc_ema(c, w):  return c.ewm(span=w, adjust=False).mean()
def calc_bb(c, w=20, n=2):
    sma = calc_sma(c, w); std = c.rolling(w).std()
    return sma + n*std, sma, sma - n*std
def calc_rsi(c, w=14):
    d = c.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(com=w-1, min_periods=w).mean(); al = l.ewm(com=w-1, min_periods=w).mean()
    return 100 - 100/(1 + ag/al.replace(0, np.nan))
def calc_macd(c, f=12, s=26, sg=9):
    ml = calc_ema(c,f) - calc_ema(c,s); sl = ml.ewm(span=sg, adjust=False).mean()
    return ml, sl, ml-sl
def calc_atr(h, l, c, w=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=w-1, min_periods=w).mean()


# ══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def _sf(v, d=4):
    try: x = float(v); return None if np.isnan(x) else round(x, d)
    except: return None


def build_analysis_payload(ticker, period, name, df):
    c = df["Close"].squeeze().dropna()
    h = df["High"].squeeze(); lo = df["Low"].squeeze()
    op = df["Open"].squeeze()
    vol = df["Volume"].squeeze() if "Volume" in df.columns else None
    n = len(c)
    cur = _sf(c.iloc[-1]); prev = _sf(c.iloc[-2]) if n > 1 else cur
    currency = "INR" if ticker.upper().endswith((".NS",".BO")) else "USD"

    hi52 = _sf(c.tail(252).max()); lo52 = _sf(c.tail(252).min())
    macd_d = {}
    if n >= 27:
        ml, sl, hl = calc_macd(c)
        macd_d = {"macd": _sf(ml.iloc[-1]), "signal": _sf(sl.iloc[-1]),
                  "histogram": _sf(hl.iloc[-1]), "hist_prev": _sf(hl.iloc[-2]) if n > 27 else None,
                  "crossover": "bullish" if (hl.iloc[-1] > 0 and hl.iloc[-2] < 0) else
                               "bearish" if (hl.iloc[-1] < 0 and hl.iloc[-2] > 0) else "none"}
    bb_d = {}
    if n >= 20:
        bbu, bbm, bbl = calc_bb(c)
        bb_d = {"upper": _sf(bbu.iloc[-1]), "mid": _sf(bbm.iloc[-1]), "lower": _sf(bbl.iloc[-1]),
                "percent_b": _sf((cur - _sf(bbl.iloc[-1])) / (_sf(bbu.iloc[-1]) - _sf(bbl.iloc[-1]))) if _sf(bbu.iloc[-1]) != _sf(bbl.iloc[-1]) else None,
                "bandwidth": _sf(((bbu.iloc[-1]-bbl.iloc[-1])/bbm.iloc[-1])*100)}
    sma20 = _sf(calc_sma(c,20).iloc[-1]) if n>=20 else None
    sma50 = _sf(calc_sma(c,50).iloc[-1]) if n>=50 else None
    sma200= _sf(calc_sma(c,200).iloc[-1]) if n>=200 else None
    rsi_v = _sf(calc_rsi(c).iloc[-1]) if n>=15 else None
    atr_v = _sf(calc_atr(h,lo,c).iloc[-1]) if n>=15 else None
    vol_d = {}
    if vol is not None:
        avg20 = _sf(vol.tail(20).mean()); cv = _sf(vol.iloc[-1])
        vol_d = {"latest": cv, "avg_20d": avg20, "ratio_vs_avg": _sf(cv/avg20) if avg20 else None}
    trend = []
    if sma20 and cur: trend.append("above_sma20" if cur>sma20 else "below_sma20")
    if sma50 and cur: trend.append("above_sma50" if cur>sma50 else "below_sma50")
    if sma200 and cur: trend.append("above_sma200" if cur>sma200 else "below_sma200")
    if sma20 and sma50: trend.append("golden_cross" if sma20>sma50 else "death_cross")

    recent = df.tail(30).copy(); recent.index = recent.index.astype(str)
    ohlcv = [{"date": d[:10], "open": _sf(r.get("Open")), "high": _sf(r.get("High")),
               "low": _sf(r.get("Low")), "close": _sf(r.get("Close")),
               "volume": int(r["Volume"]) if "Volume" in r and pd.notna(r["Volume"]) else None}
              for d, r in recent.iterrows()]
    return {
        "ticker": ticker, "name": name, "currency": currency, "period": period, "bars": n,
        "price": {"current": cur, "prev": prev, "change": _sf(cur-prev) if cur and prev else None,
                  "change_pct": _sf(((cur-prev)/prev)*100) if cur and prev else None,
                  "52w_high": hi52, "52w_low": lo52,
                  "pct_from_52h": _sf(((cur-hi52)/hi52)*100) if cur and hi52 else None},
        "ma": {"sma20": sma20, "sma50": sma50, "sma200": sma200,
               "ema9": _sf(calc_ema(c,9).iloc[-1]), "ema21": _sf(calc_ema(c,21).iloc[-1])},
        "bb": bb_d, "rsi": {"value": rsi_v, "last5": [_sf(v) for v in calc_rsi(c).tail(5).tolist()] if n>=20 else []},
        "macd": macd_d, "atr": {"value": atr_v, "pct": _sf((atr_v/cur)*100) if atr_v and cur else None},
        "volume": vol_d, "trend": trend, "ohlcv": ohlcv,
    }


def build_prompt(payload):
    p = payload; px = p["price"]; ma = p["ma"]; bb = p.get("bb",{}); rsi = p.get("rsi",{})
    macd = p.get("macd",{}); atr = p.get("atr",{}); vol = p.get("volume",{})
    f = lambda v,d=2: f"{v:.{d}f}" if v is not None else "N/A"
    up = lambda v: ("↑ Price above" if px["current"] and v and px["current"]>v else "↓ Price below") if v else "N/A"
    lines = [
        f"You are a senior quantitative stock analyst with deep knowledge of technical analysis, market microstructure, and fundamental analysis.",
        f"Analyse the following comprehensive data for **{p['name']} ({p['ticker']})** over the {p['period']} period.",
        "",
        "## PRICE SNAPSHOT",
        f"- Current: {p['currency']} {f(px['current'])}  |  Prev Close: {p['currency']} {f(px['prev'])}",
        f"- Change: {f(px['change'])} ({f(px['change_pct'])}%)",
        f"- 52W High: {p['currency']} {f(px['52w_high'])}  |  52W Low: {p['currency']} {f(px['52w_low'])}",
        f"- Distance from 52W High: {f(px['pct_from_52h'])}%",
        "",
        "## MOVING AVERAGES",
        f"- SMA 20:  {p['currency']} {f(ma['sma20'])}  ({up(ma['sma20'])} SMA20)",
        f"- SMA 50:  {p['currency']} {f(ma['sma50'])}  ({up(ma['sma50'])} SMA50)",
        f"- SMA 200: {p['currency']} {f(ma['sma200'])}  ({up(ma['sma200'])} SMA200)",
        f"- EMA 9:   {p['currency']} {f(ma['ema9'])}  |  EMA 21: {p['currency']} {f(ma['ema21'])}",
        f"- Trend signals: {', '.join(p['trend']) or 'none'}",
        "",
        "## BOLLINGER BANDS (20,2σ)",
        f"- Upper: {f(bb.get('upper'))}  Mid: {f(bb.get('mid'))}  Lower: {f(bb.get('lower'))}",
        f"- %B: {f(bb.get('percent_b'),3)} (>1=overbought, <0=oversold)  |  Bandwidth: {f(bb.get('bandwidth'))}%",
        "",
        "## RSI (14)",
        f"- Current: {f(rsi.get('value'))}  Zone: {'OVERBOUGHT' if rsi.get('value') and rsi['value']>70 else 'OVERSOLD' if rsi.get('value') and rsi['value']<30 else 'NEUTRAL'}",
        f"- Last 5: {', '.join(f(v) for v in rsi.get('last5',[]))}",
        "",
        "## MACD (12,26,9)",
        f"- MACD: {f(macd.get('macd'))}  Signal: {f(macd.get('signal'))}  Histogram: {f(macd.get('histogram'))} (prev: {f(macd.get('hist_prev'))})",
        f"- Crossover: {(macd.get('crossover') or 'none').upper()}",
        "",
        "## VOLATILITY & VOLUME",
        f"- ATR(14): {p['currency']} {f(atr.get('value'))} ({f(atr.get('pct'))}% of price)",
        f"- Latest Vol: {int(vol['latest']) if vol.get('latest') else 'N/A'}  |  20D Avg: {int(vol['avg_20d']) if vol.get('avg_20d') else 'N/A'}  |  Ratio: {f(vol.get('ratio_vs_avg'))}x",
        "",
        "## RECENT OHLCV (last 30 trading days)",
        "date,open,high,low,close,volume",
    ] + [f"{r['date']},{r['open']},{r['high']},{r['low']},{r['close']},{r['volume']}" for r in p["ohlcv"]] + [
        "",
        "---",
        "## INSTRUCTIONS",
        "Based on ALL data above plus your knowledge of current macro/sector/news context for this stock:",
        "",
        "Respond with a single valid JSON object. No markdown. No extra text. Exact structure:",
        '{"verdict":"BUY|SELL|HOLD","confidence":"Low|Medium|High","time_horizon":"Short|Mid|Long",',
        '"price_targets":{"entry":0.0,"stop_loss":0.0,"target_1":0.0,"target_2":0.0},',
        '"technical_analysis":"Detailed multi-paragraph technical breakdown. Which indicators agree/conflict. Key levels.",',
        '"news_and_macro":"What you know about recent news, earnings, macro environment, sector trends that affect this stock.",',
        '"risk_factors":"Key risks that could invalidate this trade call.",',
        '"action_plan":"Step-by-step concrete action for a trader right now. Entry timing, position sizing guidance, exit rules.",',
        '"summary":"One clear sentence with the core thesis."}',
    ]
    return "\n".join(lines)


def call_openrouter(model_id, prompt):
    if not OPEN_ROUTER_API_KEY:
        raise ValueError("OPEN_ROUTER_API_KEY environment variable is not set.")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://starfish.finance",
                 "X-Title": "Starfish Stock Analyzer"},
        json={"model": model_id, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.15, "max_tokens": 2048},
        timeout=90,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    m = re.search(r'\{.*\}', content, re.DOTALL)
    if m: content = m.group(0)
    return json.loads(content)


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════
_C = {"bg":"rgba(0,0,0,0)","paper":"rgba(0,0,0,0)","grid":"rgba(255,255,255,0.06)","axis":"#555",
      "text":"#888","white":"#fff","green":"#26a69a","red":"#ef5350",
      "sma20":"#FFD700","sma50":"#FF8C00","sma200":"#00BFFF",
      "bb_u":"rgba(120,180,255,0.7)","bb_l":"rgba(120,180,255,0.7)","bb_f":"rgba(120,180,255,0.06)",
      "rsi":"#a78bfa","rsi_ob":"rgba(239,83,80,0.25)","rsi_os":"rgba(38,166,154,0.25)",
      "macd":"#60a5fa","sig":"#f97316","hp":"rgba(38,166,154,0.8)","hn":"rgba(239,83,80,0.8)",
      "vu":"rgba(38,166,154,0.5)","vd":"rgba(239,83,80,0.5)"}


def build_chart(ticker, period, chart_type, indicators):
    data, err = fetch_yfinance_data(ticker, period)
    if err: return None, f"Data error: {err}"
    if data is None or data.empty: return None, f"No data for '{ticker}'. Use .NS for NSE stocks."
    missing = {"Open","High","Low","Close"} - set(data.columns)
    if missing: return None, f"Missing: {missing}"
    data = data.dropna(subset=["Close"])
    if len(data) < 5: return None, "Not enough data points."

    cl = data["Close"].squeeze(); hi = data["High"].squeeze()
    lo = data["Low"].squeeze(); op = data["Open"].squeeze()
    vol = data["Volume"].squeeze() if "Volume" in data.columns else None
    dates = data.index; name = _get_name(ticker)
    currency = "INR" if ticker.upper().endswith((".NS",".BO")) else "USD"

    sv = "vol" in indicators and vol is not None
    sr = "rsi" in indicators; sm = "macd" in indicators
    rows = 1 + int(sv) + int(sr) + int(sm)
    rh = {1:[1.0],2:[0.65,0.35],3:[0.55,0.22,0.23],4:[0.50,0.17,0.17,0.16]}.get(rows,[0.5,0.17,0.17,0.16])
    titles = [f"{name} ({ticker.upper()})"]
    if sv: titles.append("Volume")
    if sr: titles.append("RSI (14)")
    if sm: titles.append("MACD (12, 26, 9)")
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=rh, subplot_titles=titles)
    rv = 2 if sv else None; rr = (2+int(sv)) if sr else None; rm = (2+int(sv)+int(sr)) if sm else None

    if chart_type == "candlestick":
        fig.add_trace(go.Candlestick(x=dates,open=op,high=hi,low=lo,close=cl,name="Price",
            increasing_line_color=_C["green"],increasing_fillcolor="rgba(38,166,154,.18)",
            decreasing_line_color=_C["red"],decreasing_fillcolor="rgba(239,83,80,.18)",
            line=dict(width=1)), row=1,col=1)
    else:
        fig.add_trace(go.Scatter(x=dates,y=cl,mode="lines",name="Price",
            line=dict(color=_C["white"],width=2),fill="tozeroy",fillcolor="rgba(255,255,255,.04)"),row=1,col=1)

    if "sma" in indicators:
        for w,color,lbl in [(20,_C["sma20"],"SMA 20"),(50,_C["sma50"],"SMA 50"),(200,_C["sma200"],"SMA 200")]:
            if len(cl) >= w:
                fig.add_trace(go.Scatter(x=dates,y=calc_sma(cl,w),mode="lines",name=lbl,
                    line=dict(color=color,width=1.2),opacity=0.85),row=1,col=1)
    if "bb" in indicators and len(cl) >= 20:
        bbu,bbm,bbl = calc_bb(cl)
        fig.add_trace(go.Scatter(x=dates,y=bbu,mode="lines",name="BB Upper",
            line=dict(color=_C["bb_u"],width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=dates,y=bbl,mode="lines",name="BB Lower",
            line=dict(color=_C["bb_l"],width=1,dash="dot"),
            fill="tonexty",fillcolor=_C["bb_f"]),row=1,col=1)
    if sv and vol is not None:
        colors = [_C["vu"] if c>=o else _C["vd"] for c,o in zip(cl,op)]
        fig.add_trace(go.Bar(x=dates,y=vol,name="Volume",marker_color=colors,showlegend=False),row=rv,col=1)
    if sr and len(cl) >= 15:
        rv2 = calc_rsi(cl)
        fig.add_trace(go.Scatter(x=dates,y=rv2,mode="lines",name="RSI",
            line=dict(color=_C["rsi"],width=1.5),showlegend=False),row=rr,col=1)
        fig.add_hrect(y0=70,y1=100,row=rr,col=1,fillcolor=_C["rsi_ob"],line_width=0,layer="below")
        fig.add_hrect(y0=0,y1=30,row=rr,col=1,fillcolor=_C["rsi_os"],line_width=0,layer="below")
        for lvl,c in [(70,"rgba(239,83,80,.5)"),(30,"rgba(38,166,154,.5)"),(50,"rgba(255,255,255,.15)")]:
            fig.add_hline(y=lvl,row=rr,col=1,line=dict(color=c,width=0.8,dash="dash"))
    if sm and len(cl) >= 27:
        ml,sl,hl = calc_macd(cl)
        hc = [_C["hp"] if v>=0 else _C["hn"] for v in hl.fillna(0)]
        fig.add_trace(go.Bar(x=dates,y=hl,name="MACD Hist",marker_color=hc,showlegend=False),row=rm,col=1)
        fig.add_trace(go.Scatter(x=dates,y=ml,mode="lines",name="MACD",
            line=dict(color=_C["macd"],width=1.5),showlegend=False),row=rm,col=1)
        fig.add_trace(go.Scatter(x=dates,y=sl,mode="lines",name="Signal",
            line=dict(color=_C["sig"],width=1.5),showlegend=False),row=rm,col=1)
        fig.add_hline(y=0,row=rm,col=1,line=dict(color="rgba(255,255,255,.2)",width=0.8,dash="dash"))

    ax = dict(gridcolor=_C["grid"],color=_C["axis"],showline=False,zeroline=False,tickfont=dict(size=9,color=_C["text"]))
    fig.update_layout(
        height=420+120*(rows-1), plot_bgcolor=_C["bg"], paper_bgcolor=_C["paper"],
        font=dict(color=_C["text"],family="'DM Sans',sans-serif",size=11),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,
                    bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=_C["text"])),
        hovermode="x unified", margin=dict(l=55,r=20,t=55,b=30),
        hoverlabel=dict(bgcolor="rgba(12,12,12,.95)",bordercolor="rgba(255,255,255,.12)",font=dict(color="#fff")),
        xaxis_rangeslider_visible=False, dragmode="pan",
    )
    for i in range(1, rows+1):
        fig.update_layout(**{f"xaxis{'' if i==1 else i}": {**ax,"rangeslider":{"visible":False}}})
        fig.update_layout(**{f"yaxis{'' if i==1 else i}": {**ax}})
    if sr: fig.update_layout(**{f"yaxis{'' if rr==1 else rr}": {**ax,"range":[0,100]}})
    for ann in fig.layout.annotations: ann.font.color="#555"; ann.font.size=10
    return pyo.plot(fig,output_type="div",include_plotlyjs=False), None


# ══════════════════════════════════════════════════════════════════════════════
# HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_INDICATORS = {"sma","vol"}


def render_page(ticker, period, chart_type, active_indicators, graph_html, error):
    chips = "".join(
        f'<span class="{"chip active" if s==ticker else "chip"}" onclick="setTicker(\'{s}\')">{s}</span>\n'
        for s,_ in POPULAR_STOCKS)
    popts = "".join(f'<option value="{v}" {"selected" if v==period else ""}>{lbl}</option>\n' for v,lbl in PERIODS)
    ct_c  = "selected" if chart_type=="candlestick" else ""
    ct_l  = "selected" if chart_type=="line" else ""
    ichips= "".join(
        f'<span class="{"ind-chip active" if k in active_indicators else "ind-chip"}" data-ind="{k}" onclick="toggleInd(this)">{lbl}</span>\n'
        for k,lbl in INDICATORS)
    content = (f'<div class="error-box">{error}</div>' if error else
               graph_html if graph_html else '<div class="empty-state">Enter a ticker above.</div>')
    ntabs = "".join(
        f'<button class="{"news-tab active" if i==0 else "news-tab"}" data-handle="{ch["handle"]}">'
        f'{ch["label"]} <span class="news-tag">{ch["region"]}</span><span class="news-tag">{ch["lang"]}</span></button>\n'
        for i,ch in enumerate(NEWS_CHANNELS))

    # AI model cards
    ai_cards = ""
    for m in AI_MODELS:
        rl = rl_check(m["key"])
        pm = int((rl["rpm_used"]/rl["rpm_max"])*100)
        pd_ = int((rl["rpd_used"]/rl["rpd_max"])*100)
        ex  = " exhausted" if not rl["available"] else ""
        ai_cards += f"""<div class="ai-model-card{ex}" data-model="{m['id']}" data-key="{m['key']}" data-color="{m['color']}" data-label="{m['label']}" onclick="selectModel(this)">
  <div class="ai-model-hdr"><span class="ai-dot" style="background:{m['color']}"></span><span class="ai-mname">{m['label']}</span>{"" if rl['available'] else '<span class="ai-rl-badge">Rate Limited</span>'}</div>
  <div class="ai-mdesc">{m['desc']}</div>
  <div class="ai-rl-bars">
    <div class="ai-rl-row"><span class="ai-rl-lbl">RPM</span><div class="ai-bar-wrap"><div class="ai-bar" id="bar-rpm-{m['key']}" style="width:{pm}%;background:{m['color']}55;border-right:2px solid {m['color']}"></div></div><span class="ai-rl-cnt" id="rpm-{m['key']}">{rl['rpm_used']}/{rl['rpm_max']}</span></div>
    <div class="ai-rl-row"><span class="ai-rl-lbl">RPD</span><div class="ai-bar-wrap"><div class="ai-bar" id="bar-rpd-{m['key']}" style="width:{pd_}%;background:{m['color']}33;border-right:2px solid {m['color']}88"></div></div><span class="ai-rl-cnt" id="rpd-{m['key']}">{rl['rpd_used']}/{rl['rpd_max']}</span></div>
  </div>
</div>"""

    fh = NEWS_CHANNELS[0]["handle"]
    ai_js = json.dumps(list(active_indicators))
    models_js = json.dumps([{"id":m["id"],"key":m["key"],"label":m["label"],"color":m["color"]} for m in AI_MODELS])

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
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg:#060606;--sur:rgba(255,255,255,.04);--bdr:rgba(255,255,255,.09);--bds:rgba(255,255,255,.05);
          --tx:#f0f0f0;--txm:#666;--txd:#3a3a3a;--acc:#fff;--acm:rgba(255,255,255,.1);
          --blur:blur(20px);--r:16px;--rs:9px}}
    body{{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;
         -webkit-font-smoothing:antialiased;overflow-x:hidden}}
    body::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
      background:radial-gradient(ellipse 90% 55% at 15% 5%,rgba(255,255,255,.022) 0%,transparent 55%),
                 radial-gradient(ellipse 55% 45% at 85% 85%,rgba(255,255,255,.012) 0%,transparent 50%)}}
    header{{position:sticky;top:0;z-index:100;height:58px;display:flex;align-items:center;
            justify-content:space-between;padding:0 28px;background:rgba(6,6,6,.75);
            backdrop-filter:var(--blur);border-bottom:1px solid var(--bds)}}
    .logo{{display:flex;align-items:center;gap:10px;font-size:.9rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--acc)}}
    .logo-pip{{width:7px;height:7px;border-radius:50%;background:var(--acc);animation:blink 2.8s ease-in-out infinite}}
    @keyframes blink{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.2;transform:scale(.65)}}}}
    .subtitle{{font-size:.72rem;color:var(--txd);letter-spacing:.03em}}
    main{{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:30px 20px 64px}}
    .glass{{background:var(--sur);backdrop-filter:var(--blur);border:1px solid var(--bdr);border-radius:var(--r)}}
    .panel{{padding:26px 30px;margin-bottom:18px}}
    .panel-label{{font-size:.62rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--txd);margin-bottom:20px}}
    form{{display:grid;grid-template-columns:1.5fr 1fr 1fr auto;gap:14px;align-items:end}}
    .fg label{{display:block;font-size:.7rem;font-weight:500;letter-spacing:.05em;color:var(--txm);margin-bottom:8px}}
    input,select{{width:100%;background:rgba(255,255,255,.035);border:1px solid var(--bdr);border-radius:var(--rs);
                  color:var(--tx);padding:10px 14px;font-size:.875rem;font-family:inherit;outline:none;
                  transition:border-color .2s,background .2s,box-shadow .2s;appearance:none;-webkit-appearance:none}}
    input::placeholder{{color:var(--txd)}}
    input:focus,select:focus{{border-color:rgba(255,255,255,.28);background:rgba(255,255,255,.065);box-shadow:0 0 0 3px rgba(255,255,255,.05)}}
    select{{cursor:pointer;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 13px center;padding-right:34px}}
    select option{{background:#111;color:#f0f0f0}}
    .btn{{background:var(--acc);color:#000;border:none;border-radius:var(--rs);padding:10px 26px;
          font-size:.8rem;font-weight:600;font-family:inherit;cursor:pointer;white-space:nowrap;
          letter-spacing:.09em;text-transform:uppercase;transition:opacity .18s,transform .13s;height:42px}}
    .btn:hover{{opacity:.85}}.btn:active{{transform:scale(.96)}}
    .chips{{display:flex;flex-wrap:wrap;gap:7px;margin-top:22px;padding-top:20px;border-top:1px solid var(--bds)}}
    .chip{{background:transparent;border:1px solid var(--bdr);border-radius:100px;padding:5px 15px;
           font-size:.72rem;font-family:'DM Mono',monospace;cursor:pointer;color:var(--txm);
           letter-spacing:.05em;transition:all .16s;user-select:none}}
    .chip:hover{{border-color:rgba(255,255,255,.3);color:var(--tx);background:var(--acm)}}
    .chip.active{{background:var(--acc);border-color:var(--acc);color:#000;font-weight:600}}
    .ind-row{{display:flex;flex-wrap:wrap;gap:7px;margin-top:16px;padding-top:16px;border-top:1px solid var(--bds);align-items:center}}
    .ind-label{{font-size:.62rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txd);margin-right:4px}}
    .ind-chip{{background:transparent;border:1px solid var(--bdr);border-radius:100px;padding:4px 14px;
               font-size:.7rem;font-family:'DM Mono',monospace;cursor:pointer;color:var(--txm);
               letter-spacing:.05em;transition:all .16s;user-select:none}}
    .ind-chip:hover{{border-color:rgba(255,255,255,.3);color:var(--tx);background:var(--acm)}}
    .ind-chip.active{{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.35);color:var(--tx);font-weight:600}}
    .chart-card{{padding:20px 16px 12px;min-height:460px;display:flex;align-items:flex-start;justify-content:center;overflow:hidden}}
    .chart-card>div{{width:100%}}
    .error-box{{border:1px solid rgba(255,255,255,.1);border-left:3px solid rgba(255,255,255,.45);border-radius:var(--rs);padding:16px 20px;color:#999;font-size:.875rem;background:rgba(255,255,255,.025);width:100%;line-height:1.6}}
    .empty-state{{color:var(--txd);font-size:.85rem;text-align:center;letter-spacing:.03em}}

    /* ── AI Panel ── */
    .ai-panel{{padding:26px 30px;margin-top:18px}}
    .ai-models-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}}
    .ai-model-card{{background:rgba(255,255,255,.03);border:1px solid var(--bdr);border-radius:12px;
                    padding:16px;cursor:pointer;transition:all .2s;user-select:none}}
    .ai-model-card:hover:not(.exhausted){{border-color:rgba(255,255,255,.22);background:rgba(255,255,255,.06)}}
    .ai-model-card.selected{{border-color:rgba(255,255,255,.4);background:rgba(255,255,255,.07);box-shadow:0 0 0 1px rgba(255,255,255,.1)}}
    .ai-model-card.exhausted{{opacity:.45;cursor:not-allowed}}
    .ai-model-hdr{{display:flex;align-items:center;gap:8px;margin-bottom:4px}}
    .ai-dot{{width:7px;height:7px;border-radius:50%;flex-shrink:0}}
    .ai-mname{{font-size:.8rem;font-weight:600;color:var(--tx)}}
    .ai-mdesc{{font-size:.67rem;color:var(--txm);margin-bottom:12px}}
    .ai-rl-bars{{display:flex;flex-direction:column;gap:5px}}
    .ai-rl-row{{display:flex;align-items:center;gap:6px}}
    .ai-rl-lbl{{font-size:.58rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--txd);width:26px}}
    .ai-bar-wrap{{flex:1;height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}}
    .ai-bar{{height:100%;border-radius:2px;transition:width .5s ease}}
    .ai-rl-cnt{{font-size:.58rem;font-family:'DM Mono',monospace;color:var(--txm);width:34px;text-align:right}}
    .ai-rl-badge{{font-size:.55rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
                  padding:2px 7px;border-radius:100px;margin-left:auto;
                  background:rgba(255,160,0,.1);border:1px solid rgba(255,160,0,.25);color:#ffaa44}}
    .ai-action-row{{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
    .btn-ai{{background:linear-gradient(135deg,rgba(255,255,255,.1),rgba(255,255,255,.05));
             border:1px solid rgba(255,255,255,.18);border-radius:var(--rs);color:var(--tx);
             padding:10px 28px;font-size:.8rem;font-weight:600;font-family:inherit;cursor:pointer;
             letter-spacing:.09em;text-transform:uppercase;transition:all .18s}}
    .btn-ai:hover{{background:rgba(255,255,255,.14);border-color:rgba(255,255,255,.32)}}
    .btn-ai:active{{transform:scale(.96)}}.btn-ai:disabled{{opacity:.35;cursor:not-allowed;transform:none}}
    .ai-sel-label{{font-size:.72rem;color:var(--txm)}}
    .ai-timer{{font-size:.67rem;font-family:'DM Mono',monospace;color:var(--txd);margin-left:auto}}

    /* result */
    .ai-result{{display:none;border:1px solid var(--bdr);border-radius:12px;overflow:hidden;margin-top:4px}}
    .ai-result.show{{display:block}}
    .ai-verdict-bar{{display:flex;align-items:center;gap:14px;padding:18px 22px;border-bottom:1px solid var(--bds);flex-wrap:wrap;gap:12px}}
    .ai-badge{{font-size:.95rem;font-weight:700;letter-spacing:.12em;padding:8px 20px;border-radius:8px;text-transform:uppercase;flex-shrink:0}}
    .v-BUY{{background:rgba(38,166,154,.2);border:1px solid rgba(38,166,154,.5);color:#26a69a}}
    .v-SELL{{background:rgba(239,83,80,.2);border:1px solid rgba(239,83,80,.5);color:#ef5350}}
    .v-HOLD{{background:rgba(255,196,0,.15);border:1px solid rgba(255,196,0,.4);color:#ffc400}}
    .ai-vmeta{{display:flex;flex-direction:column;gap:4px;flex:1}}
    .ai-summary{{font-size:.84rem;color:var(--tx);line-height:1.5}}
    .ai-meta-row{{display:flex;gap:14px;flex-wrap:wrap}}
    .ai-mi{{font-size:.67rem;color:var(--txm)}}.ai-mi strong{{color:var(--txd)}}
    .ai-model-tag{{display:inline-flex;align-items:center;gap:5px;font-size:.6rem;font-weight:600;
                   letter-spacing:.08em;text-transform:uppercase;padding:3px 9px;border-radius:100px;
                   border:1px solid var(--bdr);color:var(--txm);background:rgba(255,255,255,.04);white-space:nowrap}}
    .ai-pts{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--bds);border-bottom:1px solid var(--bds)}}
    .ai-pt{{background:var(--bg);padding:14px 16px;text-align:center}}
    .ai-pt-lbl{{font-size:.58rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--txd);margin-bottom:5px}}
    .ai-pt-val{{font-size:.92rem;font-weight:600;font-family:'DM Mono',monospace}}
    .pt-e{{color:#fff}}.pt-sl{{color:#ef5350}}.pt-t1{{color:#26a69a}}.pt-t2{{color:#00BFFF}}
    .ai-secs{{padding:0}}
    .ai-sec{{padding:18px 22px;border-bottom:1px solid var(--bds)}}
    .ai-sec:last-child{{border-bottom:none}}
    .ai-sec-hdr{{font-size:.58rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
                 color:var(--txd);margin-bottom:10px;display:flex;align-items:center;gap:7px}}
    .ai-sec-body{{font-size:.82rem;color:#bbb;line-height:1.8;white-space:pre-wrap;word-break:break-word}}
    .ai-loading{{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:50px 24px;gap:14px}}
    .ai-spin{{width:26px;height:26px;border-radius:50%;border:2px solid rgba(255,255,255,.08);
              border-top-color:rgba(255,255,255,.5);animation:spin .7s linear infinite}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}
    .ai-load-txt{{font-size:.78rem;color:var(--txm);letter-spacing:.04em}}
    .ai-err{{padding:20px 22px;color:#ff9966;font-size:.82rem;line-height:1.6}}

    /* News */
    .news-panel{{padding:26px 30px;margin-top:18px}}
    .news-live-dot{{display:inline-block;width:6px;height:6px;border-radius:50%;background:#ff4444;
                    margin-right:6px;animation:lp 1.4s ease-in-out infinite;vertical-align:middle}}
    @keyframes lp{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(.6)}}}}
    .news-tabs{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap}}
    .news-tab{{background:transparent;border:1px solid var(--bdr);border-radius:100px;padding:6px 18px;
               font-size:.72rem;font-family:'DM Mono',monospace;cursor:pointer;color:var(--txm);
               letter-spacing:.05em;transition:all .16s;user-select:none}}
    .news-tab:hover{{border-color:rgba(255,255,255,.3);color:var(--tx);background:var(--acm)}}
    .news-tab.active{{background:var(--acc);border-color:var(--acc);color:#000;font-weight:600}}
    .news-tab.active .news-tag{{background:rgba(0,0,0,.15);color:rgba(0,0,0,.5)}}
    .news-iframe-wrap{{position:relative;width:100%;padding-top:56.25%;border-radius:var(--rs);overflow:hidden;background:rgba(0,0,0,.5)}}
    .news-loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                   color:var(--txm);font-size:.8rem;letter-spacing:.05em;flex-direction:column;gap:12px}}
    .news-spinner{{width:22px;height:22px;border-radius:50%;border:2px solid rgba(255,255,255,.1);border-top-color:rgba(255,255,255,.5);animation:spin .8s linear infinite}}
    .news-iframe-wrap iframe{{position:absolute;inset:0;width:100%;height:100%;border:none}}
    .news-tag{{font-size:.55rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
               padding:1px 5px;border-radius:4px;background:rgba(255,255,255,.08);color:var(--txd);margin-left:4px;vertical-align:middle}}
    .nsb{{display:inline-flex;align-items:center;gap:5px;font-size:.62rem;font-weight:700;
           letter-spacing:.1em;text-transform:uppercase;padding:3px 10px;border-radius:100px;white-space:nowrap}}
    .nsb.live{{background:rgba(255,60,60,.15);border:1px solid rgba(255,60,60,.35);color:#ff6b6b}}
    .nsb.live::before{{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#ff4444;animation:lp 1.4s ease-in-out infinite}}
    .nsb.latest{{background:rgba(255,255,255,.06);border:1px solid var(--bdr);color:var(--txm)}}
    .nsb.error{{background:rgba(255,160,0,.1);border:1px solid rgba(255,160,0,.25);color:#ffaa44}}
    @media(max-width:860px){{form{{grid-template-columns:1fr 1fr;gap:12px}}.fg:first-child{{grid-column:span 2}}.btn{{grid-column:span 2;width:100%}}.ai-models-grid{{grid-template-columns:1fr}}.ai-pts{{grid-template-columns:repeat(2,1fr)}}}}
    @media(max-width:600px){{header{{padding:0 16px}}.subtitle{{display:none}}main{{padding:18px 14px 48px}}.panel{{padding:20px 18px}}.chart-card{{padding:16px 10px 10px;min-height:300px}}.news-panel,.ai-panel{{padding:20px 18px}}}}
  </style>
</head>
<body>
<header>
  <div class="logo"><span class="logo-pip"></span>Starfish</div>
  <span class="subtitle">LIVE MARKETS · AI ANALYSIS</span>
</header>
<main>

<div class="glass panel">
  <div class="panel-label">Search</div>
  <form method="POST" action="/" id="main-form">
    <input type="hidden" name="indicators" id="inds-h" value="{','.join(active_indicators)}"/>
    <div class="fg">
      <label for="ticker">Ticker Symbol</label>
      <input id="ticker" name="ticker" type="text" value="{ticker}"
             placeholder="AAPL, GOOGL, TCS.NS" required autocomplete="off" autocapitalize="characters" spellcheck="false"/>
    </div>
    <div class="fg">
      <label for="period">Time Range</label>
      <select id="period" name="period">{popts}</select>
    </div>
    <div class="fg">
      <label for="chart_type">Chart Type</label>
      <select id="chart_type" name="chart_type">
        <option value="candlestick" {ct_c}>Candlestick</option>
        <option value="line" {ct_l}>Line</option>
      </select>
    </div>
    <button type="submit" class="btn">Load</button>
  </form>
  <div class="chips">{chips}</div>
  <div class="ind-row"><span class="ind-label">Indicators</span>{ichips}</div>
</div>

<div class="glass chart-card">{content}</div>

<!-- ── AI Analysis ── -->
<div class="glass ai-panel">
  <div class="panel-label">&#x2728;&nbsp; AI Trading Analysis</div>
  <div class="ai-models-grid" id="ai-grid">{ai_cards}</div>
  <div class="ai-action-row">
    <button class="btn-ai" id="btn-ai" onclick="runAnalysis()" disabled>Analyse&nbsp;{ticker}</button>
    <span class="ai-sel-label" id="ai-sel-lbl">&#x2190; Select a model</span>
    <span class="ai-timer" id="ai-timer"></span>
  </div>
  <div class="ai-result" id="ai-result"></div>
</div>

<!-- ── News ── -->
<div class="glass news-panel">
  <div class="panel-label"><span class="news-live-dot"></span>Live Financial News</div>
  <div class="news-tabs" id="ntabs">{ntabs}</div>
  <div class="news-iframe-wrap">
    <div id="nload" class="news-loading"><div class="news-spinner"></div><span>Loading stream&hellip;</span></div>
    <iframe id="nframe" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="display:none"></iframe>
  </div>
  <div style="margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span id="nbadge" class="nsb" style="display:none"></span>
  </div>
</div>

</main>
<script>
var TICKER = {json.dumps(ticker)};
var PERIOD = {json.dumps(period)};
var MODELS = {models_js};

// Chips
function setTicker(s){{document.getElementById('ticker').value=s;document.getElementById('main-form').submit();}}

// Indicator toggles
var aInds = {ai_js};
function toggleInd(el){{
  var k=el.dataset.ind,i=aInds.indexOf(k);
  i===-1?(aInds.push(k),el.classList.add('active')):(aInds.splice(i,1),el.classList.remove('active'));
  document.getElementById('inds-h').value=aInds.join(',');
  document.getElementById('main-form').submit();
}}

// AI model selection
var selModelId=null,selModelKey=null,timerIv=null;

function selectModel(card){{
  if(card.classList.contains('exhausted'))return;
  document.querySelectorAll('.ai-model-card').forEach(c=>c.classList.remove('selected'));
  card.classList.add('selected');
  selModelId=card.dataset.model; selModelKey=card.dataset.key;
  document.getElementById('ai-sel-lbl').textContent='Model: '+card.dataset.label;
  document.getElementById('btn-ai').disabled=false;
  if(timerIv)clearInterval(timerIv);
  refreshRateLimits();
  timerIv=setInterval(refreshRateLimits,4000);
}}

function refreshRateLimits(){{
  fetch('/api/rate-limits').then(r=>r.json()).then(data=>{{
    MODELS.forEach(m=>{{
      var d=data[m.key]; if(!d)return;
      var rpmEl=document.getElementById('rpm-'+m.key);
      var rpdEl=document.getElementById('rpd-'+m.key);
      var barRpm=document.getElementById('bar-rpm-'+m.key);
      var barRpd=document.getElementById('bar-rpd-'+m.key);
      if(rpmEl)rpmEl.textContent=d.rpm_used+'/'+d.rpm_max;
      if(rpdEl)rpdEl.textContent=d.rpd_used+'/'+d.rpd_max;
      if(barRpm)barRpm.style.width=Math.round((d.rpm_used/d.rpm_max)*100)+'%';
      if(barRpd)barRpd.style.width=Math.round((d.rpd_used/d.rpd_max)*100)+'%';
    }});
    if(selModelKey&&data[selModelKey]){{
      var sd=data[selModelKey];
      var timerEl=document.getElementById('ai-timer');
      if(sd.rpm_used>=sd.rpm_max){{
        timerEl.textContent='RPM full — resets in '+sd.rpm_reset_secs+'s';
        document.getElementById('btn-ai').disabled=true;
      }}else{{
        timerEl.textContent='RPM: '+sd.rpm_used+'/'+sd.rpm_max+' used  ·  RPD: '+sd.rpd_used+'/'+sd.rpd_max+' used';
        document.getElementById('btn-ai').disabled=false;
      }}
    }}
  }}).catch(()=>{{}});
}}

// Run analysis
function runAnalysis(){{
  if(!selModelId)return;
  var btn=document.getElementById('btn-ai');
  var res=document.getElementById('ai-result');
  btn.disabled=true; btn.textContent='Analysing\u2026';
  res.className='ai-result show';
  res.innerHTML='<div class="ai-loading"><div class="ai-spin"></div><div class="ai-load-txt">Crunching '+TICKER+' data with AI\u2026 (20\u201340s)</div></div>';

  fetch('/api/ai-analysis',{{
    method:'POST', headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{ticker:TICKER,period:PERIOD,model_id:selModelId}})
  }}).then(r=>r.json()).then(data=>{{
    btn.disabled=false; btn.textContent='Analyse '+TICKER;
    if(data.error){{res.innerHTML='<div class="ai-err">&#x26A0; '+esc(data.error)+'</div>';return;}}
    renderResult(data); refreshRateLimits();
  }}).catch(err=>{{
    btn.disabled=false; btn.textContent='Analyse '+TICKER;
    res.innerHTML='<div class="ai-err">&#x26A0; Network error: '+esc(String(err))+'</div>';
  }});
}}

function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
function fn(v,d){{d=d||2;return(v==null||v===undefined)?'N/A':Number(v).toFixed(d);}}

function renderResult(data){{
  var r=data.analysis;
  var m=MODELS.find(x=>x.id===data.model_id)||{{}};
  var verdict=(r.verdict||'HOLD').toUpperCase();
  var pt=r.price_targets||{{}};
  var secs=[
    {{icon:'&#x1F4CA;',lbl:'Technical Analysis',key:'technical_analysis'}},
    {{icon:'&#x1F4F0;',lbl:'News & Macro Context',key:'news_and_macro'}},
    {{icon:'&#x26A0;&#xFE0F;',lbl:'Risk Factors',key:'risk_factors'}},
    {{icon:'&#x1F3AF;',lbl:"Trader's Action Plan",key:'action_plan'}},
  ];
  var secHtml=secs.map(s=>'<div class="ai-sec"><div class="ai-sec-hdr"><span>'+s.icon+'</span>'+s.lbl+'</div><div class="ai-sec-body">'+esc(r[s.key]||'No data provided.')+'</div></div>').join('');

  document.getElementById('ai-result').innerHTML=
    '<div class="ai-verdict-bar">'+
      '<div class="ai-badge v-'+verdict+'">'+verdict+'</div>'+
      '<div class="ai-vmeta">'+
        '<div class="ai-summary">'+esc(r.summary||'')+'</div>'+
        '<div class="ai-meta-row">'+
          '<span class="ai-mi"><strong>Confidence&nbsp;</strong>'+esc(r.confidence||'Medium')+'</span>'+
          '<span class="ai-mi"><strong>Horizon&nbsp;</strong>'+esc(r.time_horizon||'Mid')+'-term</span>'+
        '</div>'+
      '</div>'+
      '<span class="ai-model-tag"><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:'+(m.color||'#fff')+';margin-right:4px"></span>'+esc(m.label||data.model_id)+'</span>'+
    '</div>'+
    '<div class="ai-pts">'+
      '<div class="ai-pt"><div class="ai-pt-lbl">Entry</div><div class="ai-pt-val pt-e">'+fn(pt.entry)+'</div></div>'+
      '<div class="ai-pt"><div class="ai-pt-lbl">Stop Loss</div><div class="ai-pt-val pt-sl">'+fn(pt.stop_loss)+'</div></div>'+
      '<div class="ai-pt"><div class="ai-pt-lbl">Target 1</div><div class="ai-pt-val pt-t1">'+fn(pt.target_1)+'</div></div>'+
      '<div class="ai-pt"><div class="ai-pt-lbl">Target 2</div><div class="ai-pt-val pt-t2">'+fn(pt.target_2)+'</div></div>'+
    '</div>'+
    '<div class="ai-secs">'+secHtml+'</div>';
}}

// News
var nframe=document.getElementById('nframe'),nload=document.getElementById('nload'),
    nbadge=document.getElementById('nbadge'),curHandle=null;

function nSetLoad(m){{nframe.style.display='none';nload.innerHTML='<div class="news-spinner"></div><span>'+m+'</span>';nload.style.display='flex';nbadge.style.display='none';}}
function nSetErr(m){{nframe.style.display='none';nload.innerHTML='<span>'+m+'</span>';nload.style.display='flex';nbadge.className='nsb error';nbadge.textContent='\u26a0 Unavailable';nbadge.style.display='inline-flex';}}

function loadCh(h){{
  if(curHandle===h)return;
  curHandle=h; nSetLoad('Loading stream\u2026'); nframe.src='about:blank';
  fetch('/api/live-id?handle='+encodeURIComponent(h))
    .then(r=>{{if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}})
    .then(d=>{{
      if(h!==curHandle)return;
      if(d.error||!d.video_id){{nSetErr('Stream unavailable.');return;}}
      nframe.src='https://www.youtube.com/embed/'+d.video_id+'?autoplay=1&rel=0&modestbranding=1';
      nframe.style.display='block';nload.style.display='none';
      nbadge.style.display='inline-flex';
      nbadge.className=d.is_live?'nsb live':'nsb latest';
      nbadge.textContent=d.is_live?'LIVE':'\u25b6 Latest Video';
    }}).catch(()=>{{if(h!==curHandle)return;nSetErr('Could not load stream.');}});
}}

document.getElementById('ntabs').addEventListener('click',function(e){{
  var btn=e.target.closest('.news-tab');if(!btn)return;
  document.querySelectorAll('.news-tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');curHandle=null;loadCh(btn.dataset.handle);
}});
loadCh('{fh}');
setInterval(refreshRateLimits, 8000);
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET","POST"])
def index():
    ticker     = (request.form.get("ticker","AAPL") or "AAPL").strip().upper()
    period     = request.form.get("period","6mo")
    chart_type = request.form.get("chart_type","candlestick")
    ind_raw    = request.form.get("indicators",",".join(DEFAULT_INDICATORS))
    if period not in VALID_PERIODS: period = "6mo"
    if chart_type not in ("candlestick","line"): chart_type = "candlestick"
    active = set(filter(None, ind_raw.split(","))) if ind_raw else DEFAULT_INDICATORS
    graph_html, error = build_chart(ticker, period, chart_type, active)
    return render_page(ticker, period, chart_type, active, graph_html, error)


@app.route("/api/ai-analysis", methods=["POST"])
def api_ai_analysis():
    body     = request.get_json(force=True) or {}
    ticker   = (body.get("ticker","AAPL") or "AAPL").strip().upper()
    period   = body.get("period","6mo")
    model_id = (body.get("model_id") or "").strip()
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    model = next((m for m in AI_MODELS if m["id"] == model_id), None)
    if not model:
        return jsonify({"error": f"Unknown model: {model_id}"}), 400

    rl = rl_check(model["key"])
    if not rl["available"]:
        reset = rl_next_rpm_reset(model["key"])
        return jsonify({"error": f"Rate limit hit ({model['label']}): RPM {rl['rpm_used']}/{rl['rpm_max']}, RPD {rl['rpd_used']}/{rl['rpd_max']}. RPM resets in {reset}s."}), 429

    if period not in VALID_PERIODS: period = "6mo"
    df, err = fetch_yfinance_data(ticker, period)
    if err:
        return jsonify({"error": f"Data fetch failed: {err}"}), 502
    if df is None or df.empty:
        return jsonify({"error": f"No data for '{ticker}'."}), 404

    name = _get_name(ticker)
    try:
        payload = build_analysis_payload(ticker, period, name, df)
        prompt  = build_prompt(payload)
    except Exception as e:
        return jsonify({"error": f"Indicator error: {e}"}), 500

    try:
        analysis = call_openrouter(model_id, prompt)
        rl_record(model["key"])
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else 0
        if code == 429: return jsonify({"error": "OpenRouter rate limit. Wait a moment."}), 429
        return jsonify({"error": f"OpenRouter HTTP {code}: {e}"}), 502
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Model returned invalid JSON: {e}"}), 502
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"AI error: {e}"}), 500

    return jsonify({"ticker": ticker, "period": period, "model_id": model_id, "analysis": analysis})


@app.route("/api/rate-limits")
def api_rate_limits():
    return jsonify({
        m["key"]: {**rl_check(m["key"]), "rpm_reset_secs": rl_next_rpm_reset(m["key"])}
        for m in AI_MODELS
    })


@app.route("/api/live-id")
def api_live_id():
    handle = request.args.get("handle","").strip()
    if not handle: return jsonify({"error": "missing handle"}), 400
    vid, live = fetch_live_video_id(handle)
    if vid: return jsonify({"video_id": vid, "is_live": live})
    return jsonify({"error": "not found"}), 404


@app.route("/debug")
def debug():
    out, color = [], "#7fff7f"
    try:
        df, err = fetch_yfinance_data("AAPL","5d")
        if err: out.append(f"Error: {err}"); color="#ff7f7f"
        elif df is not None: out.append(f"OK shape:{df.shape}"); out.append(df.tail().to_string())
        else: out.append("No data"); color="#ffaa44"
    except Exception: out.append(traceback.format_exc()); color="#ff7f7f"
    body = "\n".join(out)
    return f"<pre style='background:#111;color:{color};padding:24px;font-family:monospace;white-space:pre-wrap'>{body}</pre>"


@app.errorhandler(500)
def e500(e):
    return f"<pre style='background:#111;color:#aaa;padding:24px;font-family:monospace'>500\n\n{traceback.format_exc()}</pre>", 500


if __name__ == "__main__":
    app.run(debug=True)
