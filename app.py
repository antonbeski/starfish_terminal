import traceback
import requests
from flask import Flask, request
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd

app = Flask(__name__)

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


def get_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    })
    return session


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_chart(ticker, period, chart_type):
    session = get_yf_session()

    try:
        data = yf.download(
            ticker, period=period, interval="1d",
            progress=False, auto_adjust=True,
            actions=False, session=session,
        )
    except Exception as e:
        return None, f"Download failed: {e}"

    if data is None or data.empty:
        return None, (
            f"No data found for '{ticker}'. "
            "Use '.NS' suffix for NSE stocks (e.g. TCS.NS)."
        )

    data = flatten_columns(data)
    missing = {"Open", "High", "Low", "Close"} - set(data.columns)
    if missing:
        return None, f"Missing columns: {missing}"

    data = data.dropna(subset=["Close"])
    if data.empty:
        return None, "No valid data after cleaning."

    try:
        t = yf.Ticker(ticker, session=session)
        name = t.fast_info.get("longName") or t.info.get("shortName") or ticker.upper()
    except Exception:
        name = ticker.upper()

    currency = "INR" if ticker.upper().endswith(".NS") else "USD"
    is_mobile_hint = True  # always emit responsive config

    # ── Color palette ──
    GREEN       = "#3ddc84"
    RED         = "#ff5f57"
    GRID        = "rgba(255,255,255,0.04)"
    AXIS_LINE   = "rgba(255,255,255,0.06)"
    TICK_COLOR  = "rgba(255,255,255,0.25)"
    HOVER_BG    = "rgba(8,8,8,0.96)"
    HOVER_BORDER= "rgba(255,255,255,0.1)"
    SPIKE_COLOR = "rgba(255,255,255,0.12)"

    if chart_type == "candlestick":
        trace = go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker.upper(),
            increasing=dict(
                line=dict(color=GREEN, width=1),
                fillcolor=GREEN,
            ),
            decreasing=dict(
                line=dict(color=RED, width=1),
                fillcolor=RED,
            ),
            whiskerwidth=0.5,
        )
    else:
        # Compute colour based on net change
        net = float(data["Close"].iloc[-1]) - float(data["Close"].iloc[0])
        line_color = GREEN if net >= 0 else RED
        fill_color = "rgba(61,220,132,0.06)" if net >= 0 else "rgba(255,95,87,0.06)"

        trace = go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=ticker.upper(),
            line=dict(color=line_color, width=1.8),
            fill="tozeroy",
            fillcolor=fill_color,
        )

    fig = go.Figure(data=trace)

    fig.update_layout(
        # Title
        title=dict(
            text=(
                f"<span style='font-family:Syne,sans-serif;font-weight:700;"
                f"font-size:15px;color:#e8e8e8'>{name}</span>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:11px;"
                f"color:#444;letter-spacing:0.1em;margin-left:10px'>{ticker.upper()}</span>"
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:10px;"
                f"color:#333;letter-spacing:0.08em;margin-left:8px'>{currency}</span>"
            ),
            x=0,
            xanchor="left",
            pad=dict(l=14, t=14),
            font=dict(size=15),
        ),

        # Backgrounds — fully transparent
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",

        # Global font
        font=dict(
            family="IBM Plex Mono, monospace",
            size=10,
            color=TICK_COLOR,
        ),

        # X Axis
        xaxis=dict(
            gridcolor=GRID,
            gridwidth=1,
            linecolor=AXIS_LINE,
            linewidth=1,
            tickcolor=AXIS_LINE,
            tickfont=dict(size=10, color=TICK_COLOR),
            rangeslider=dict(visible=False),
            showspikes=True,
            spikecolor=SPIKE_COLOR,
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            fixedrange=False,
            type="date",
        ),

        # Y Axis
        yaxis=dict(
            gridcolor=GRID,
            gridwidth=1,
            linecolor=AXIS_LINE,
            linewidth=1,
            tickcolor=AXIS_LINE,
            tickfont=dict(size=10, color=TICK_COLOR),
            side="right",
            showgrid=True,
            showspikes=True,
            spikecolor=SPIKE_COLOR,
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            fixedrange=False,
            tickformat=".2f",
        ),

        # Hover
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=HOVER_BG,
            bordercolor=HOVER_BORDER,
            font=dict(
                family="IBM Plex Mono, monospace",
                size=11,
                color="#e0e0e0",
            ),
            namelength=-1,
        ),

        # Layout
        margin=dict(l=12, r=68, t=52, b=36),
        height=460,

        # Legend (hide — title serves identification)
        showlegend=False,

        # Modebar
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(255,255,255,0.2)",
            activecolor="rgba(255,255,255,0.6)",
            orientation="v",
        ),

        # Autosize
        autosize=True,

        # Dragmode
        dragmode="zoom",
    )

    # Make the chart responsive via config
    config = dict(
        responsive=True,
        displaylogo=False,
        modeBarButtonsToRemove=[
            "select2d", "lasso2d", "autoScale2d",
            "hoverClosestCartesian", "hoverCompareCartesian",
            "toggleSpikelines",
        ],
        scrollZoom=True,
        toImageButtonOptions=dict(
            format="png", filename=f"starfish_{ticker}", scale=2
        ),
    )

    return (
        pyo.plot(fig, output_type="div", include_plotlyjs=False, config=config),
        None,
    )


# ── HTML Template ──────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover"/>
  <title>STARFISH</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ -webkit-text-size-adjust: 100%; }}

    :root {{
      --bg:        #000;
      --surface:   #0c0c0c;
      --surface-2: #111;
      --border:    #1e1e1e;
      --border-hi: #2e2e2e;
      --text:      #f0f0f0;
      --text-2:    #777;
      --text-3:    #3a3a3a;
      --green:     #3ddc84;
      --red:       #ff5f57;
      --r:         10px;
      --r-sm:      7px;
    }}

    body {{
      font-family: 'Syne', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      min-height: 100dvh;
      -webkit-font-smoothing: antialiased;
    }}

    .app {{
      display: flex;
      flex-direction: column;
      min-height: 100vh;
      min-height: 100dvh;
    }}

    /* Header */
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      background: rgba(0,0,0,0.9);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
    }}

    .header-left {{ display: flex; align-items: center; gap: 10px; }}

    .logo {{
      font-size: 1rem;
      font-weight: 800;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: var(--text);
    }}

    .logo-tag {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.58rem;
      color: var(--text-3);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      border: 1px solid var(--border);
      padding: 2px 7px;
      border-radius: 4px;
    }}

    .header-right {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.58rem;
      color: var(--text-3);
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}

    /* Content */
    .content {{
      flex: 1;
      padding: 20px 24px 40px;
      width: 100%;
      max-width: 1400px;
      margin: 0 auto;
    }}

    /* Controls */
    .controls {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      padding: 18px 20px;
      margin-bottom: 14px;
    }}

    .form-grid {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr auto;
      gap: 10px;
      align-items: end;
    }}

    .field {{ display: flex; flex-direction: column; gap: 6px; }}

    .field-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.58rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--text-3);
    }}

    input, select {{
      background: var(--surface-2);
      border: 1px solid var(--border);
      border-radius: var(--r-sm);
      color: var(--text);
      padding: 9px 12px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.82rem;
      outline: none;
      width: 100%;
      transition: border-color 0.15s, background 0.15s;
      -webkit-appearance: none;
      appearance: none;
      height: 38px;
    }}

    input::placeholder {{ color: var(--text-3); }}

    input:focus, select:focus {{
      border-color: var(--border-hi);
      background: #161616;
    }}

    .select-wrap {{ position: relative; }}

    .select-wrap::after {{
      content: '';
      position: absolute;
      right: 11px;
      top: 50%;
      transform: translateY(-50%);
      pointer-events: none;
      border-left: 4px solid transparent;
      border-right: 4px solid transparent;
      border-top: 4px solid var(--text-3);
    }}

    select {{ padding-right: 28px; cursor: pointer; }}
    select option {{ background: #111; color: #eee; }}

    .btn-wrap {{ display: flex; }}

    .btn {{
      background: var(--text);
      color: var(--bg);
      border: none;
      border-radius: var(--r-sm);
      padding: 0 20px;
      font-family: 'Syne', sans-serif;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
      white-space: nowrap;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 100%;
    }}

    .btn:hover {{ opacity: 0.85; }}
    .btn:active {{ transform: scale(0.97); opacity: 1; }}

    /* Chips */
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
    }}

    .chip {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.67rem;
      padding: 4px 11px;
      border-radius: 100px;
      border: 1px solid var(--border);
      color: var(--text-2);
      background: transparent;
      cursor: pointer;
      transition: border-color 0.12s, color 0.12s, background 0.12s;
      user-select: none;
    }}

    .chip:hover {{ border-color: var(--border-hi); color: var(--text); }}
    .chip.active {{ background: var(--text); border-color: var(--text); color: var(--bg); font-weight: 500; }}

    /* Chart */
    .chart-wrap {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--r);
      overflow: hidden;
    }}

    .chart-inner {{ padding: 4px 2px 2px; }}

    /* States */
    .empty-state {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 80px 20px;
      gap: 10px;
    }}

    .empty-icon {{
      width: 44px;
      height: 44px;
      border-radius: 50%;
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 4px;
    }}

    .empty-title {{ font-size: 0.82rem; font-weight: 600; color: var(--text-2); }}

    .empty-hint {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.67rem;
      color: var(--text-3);
      text-align: center;
      max-width: 260px;
      line-height: 1.7;
    }}

    .error-state {{
      display: flex;
      align-items: flex-start;
      gap: 12px;
      padding: 16px 18px;
      margin: 16px;
      background: rgba(255,95,87,0.05);
      border: 1px solid rgba(255,95,87,0.14);
      border-radius: var(--r-sm);
    }}

    .error-state svg {{ flex-shrink: 0; margin-top: 1px; }}

    .error-msg {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      color: var(--red);
      line-height: 1.65;
    }}

    /* Animations */
    @keyframes fadeUp {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}

    .controls   {{ animation: fadeUp 0.3s ease both; }}
    .chart-wrap {{ animation: fadeUp 0.3s 0.07s ease both; }}

    /* Responsive */
    @media (max-width: 860px) {{
      .form-grid {{ grid-template-columns: 1fr 1fr; }}
      .btn-wrap {{ grid-column: span 2; }}
    }}

    @media (max-width: 540px) {{
      .header {{ padding: 15px 16px; }}
      .content {{ padding: 14px 14px 32px; }}
      .controls {{ padding: 14px; }}
      .form-grid {{ grid-template-columns: 1fr; gap: 9px; }}
      .btn-wrap {{ grid-column: span 1; }}
      .logo-tag, .header-right {{ display: none; }}
      .empty-state {{ padding: 56px 20px; }}
    }}
  </style>
</head>
<body>
<div class="app">

  <header class="header">
    <div class="header-left">
      <span class="logo">Starfish</span>
      <span class="logo-tag">Markets</span>
    </div>
    <span class="header-right">US &amp; NSE</span>
  </header>

  <div class="content">

    <div class="controls">
      <form method="POST" action="/">
        <div class="form-grid">
          <div class="field">
            <span class="field-label">Symbol</span>
            <input id="ticker" name="ticker" type="text"
              value="{ticker}"
              placeholder="AAPL, TCS.NS ..."
              required autocomplete="off" spellcheck="false"/>
          </div>

          <div class="field">
            <span class="field-label">Range</span>
            <div class="select-wrap">
              <select id="period" name="period">{period_opts}</select>
            </div>
          </div>

          <div class="field">
            <span class="field-label">Chart</span>
            <div class="select-wrap">
              <select id="chart_type" name="chart_type">
                <option value="candlestick" {ct_candle}>Candlestick</option>
                <option value="line" {ct_line}>Line</option>
              </select>
            </div>
          </div>

          <div class="btn-wrap">
            <button type="submit" class="btn">Load</button>
          </div>
        </div>

        <div class="chips">{chips}</div>
      </form>
    </div>

    <div class="chart-wrap">
      {content}
    </div>

  </div>
</div>

<script>
  function setTicker(s) {{
    document.getElementById('ticker').value = s;
    document.querySelector('form').submit();
  }}
  document.getElementById('ticker').addEventListener('input', function() {{
    var p = this.selectionStart;
    this.value = this.value.toUpperCase();
    this.setSelectionRange(p, p);
  }});
</script>
</body>
</html>"""


def render_page(ticker, period, chart_type, graph_html, error):
    chips = ""
    for sym, _ in POPULAR_STOCKS:
        a = " active" if sym == ticker else ""
        chips += f'<span class="chip{a}" onclick="setTicker(\'{sym}\')">{sym}</span>\n'

    period_opts = ""
    for val, label in PERIODS:
        sel = "selected" if val == period else ""
        period_opts += f'<option value="{val}" {sel}>{label}</option>\n'

    ct_candle = "selected" if chart_type == "candlestick" else ""
    ct_line   = "selected" if chart_type == "line" else ""

    if error:
        content = (
            '<div class="error-state">'
            '<svg width="15" height="15" viewBox="0 0 16 16" fill="none">'
            '<circle cx="8" cy="8" r="7" stroke="#ff5f57" stroke-width="1.3"/>'
            '<path d="M8 4.8v3.6M8 10.5v.7" stroke="#ff5f57" stroke-width="1.4" stroke-linecap="round"/>'
            '</svg>'
            f'<div class="error-msg">{error}</div>'
            '</div>'
        )
    elif graph_html:
        content = f'<div class="chart-inner">{graph_html}</div>'
    else:
        content = (
            '<div class="empty-state">'
            '<div class="empty-icon">'
            '<svg width="18" height="18" viewBox="0 0 24 24" fill="none">'
            '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12" stroke="#3a3a3a"'
            ' stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg></div>'
            '<div class="empty-title">No data loaded</div>'
            '<div class="empty-hint">Enter a ticker and press Load to view the chart</div>'
            '</div>'
        )

    return PAGE.format(
        ticker=ticker,
        period_opts=period_opts,
        ct_candle=ct_candle,
        ct_line=ct_line,
        chips=chips,
        content=content,
    )


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


@app.route("/debug")
def debug():
    try:
        session = get_yf_session()
        data = yf.download("AAPL", period="5d", progress=False,
                           auto_adjust=True, session=session)
        return (
            "<pre style='background:#0a0a0a;color:#3ddc84;padding:24px;"
            "font-family:monospace;font-size:13px'>"
            f"yfinance OK\nShape: {data.shape}\n\n{data.tail().to_string()}</pre>"
        )
    except Exception:
        tb = traceback.format_exc()
        return (
            "<pre style='background:#0a0a0a;color:#ff5f57;padding:24px;"
            f"font-family:monospace;font-size:13px'>yfinance FAILED\n\n{tb}</pre>"
        ), 500


@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    return (
        "<pre style='background:#0a0a0a;color:#ff5f57;padding:24px;"
        f"font-family:monospace;font-size:13px'>500 — Internal Error\n\n{tb}</pre>"
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
