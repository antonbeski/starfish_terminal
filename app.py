import traceback
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


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_chart(ticker, period, chart_type):
    try:
        data = yf.download(ticker, period=period, interval="1d",
                           progress=False, auto_adjust=True, actions=False)
    except Exception as e:
        return None, f"Download failed: {e}"

    if data is None or data.empty:
        return None, (f"No data found for '{ticker}'. "
                      "Check the symbol — use '.NS' for NSE stocks (e.g. TCS.NS).")

    data = flatten_columns(data)
    missing = {"Open", "High", "Low", "Close"} - set(data.columns)
    if missing:
        return None, f"Missing columns: {missing}"

    data = data.dropna(subset=["Close"])
    if data.empty:
        return None, "All rows were empty after cleaning."

    try:
        name = yf.Ticker(ticker).fast_info.company_name or ticker
    except Exception:
        name = ticker

    currency = "INR" if ticker.upper().endswith(".NS") else "USD"

    if chart_type == "candlestick":
        trace = go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name=ticker,
            increasing_line_color="#26a641", decreasing_line_color="#f85149",
        )
    else:
        trace = go.Scatter(
            x=data.index, y=data["Close"], mode="lines", name=ticker,
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
        )

    fig = go.Figure(data=trace)
    fig.update_layout(
        title=dict(text=f"{name} ({ticker.upper()})", font=dict(size=20, color="#e6edf3")),
        xaxis_title="Date", yaxis_title=f"Price ({currency})",
        plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#21262d", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#21262d"),
        hovermode="x unified", margin=dict(l=50, r=30, t=60, b=50),
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False), None


def render_page(ticker, period, chart_type, graph_html, error):
    # Quick-ticker chips
    chips = ""
    for sym, _ in POPULAR_STOCKS:
        active = 'class="chip active"' if sym == ticker else 'class="chip"'
        chips += f'<span {active} onclick="setTicker(\'{sym}\')">{sym}</span>\n'

    # Period options
    period_opts = ""
    for val, label in PERIODS:
        sel = "selected" if val == period else ""
        period_opts += f'<option value="{val}" {sel}>{label}</option>\n'

    # Chart type options
    ct_candle = "selected" if chart_type == "candlestick" else ""
    ct_line   = "selected" if chart_type == "line" else ""

    # Content area
    if error:
        content = f'<div class="error-box">⚠️ {error}</div>'
    elif graph_html:
        content = graph_html
    else:
        content = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>StockChart</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}}
    header{{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px}}
    .logo{{font-size:1.4rem;font-weight:700;color:#58a6ff}}
    .subtitle{{font-size:.85rem;color:#8b949e;margin-top:2px}}
    main{{max-width:1200px;margin:0 auto;padding:28px 20px}}
    .panel{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:24px}}
    .panel-title{{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin-bottom:16px}}
    form{{display:grid;grid-template-columns:1fr 1fr 1fr auto;gap:12px;align-items:end}}
    @media(max-width:700px){{form{{grid-template-columns:1fr 1fr}}.btn{{grid-column:span 2}}}}
    label{{display:block;font-size:.8rem;color:#8b949e;margin-bottom:6px;font-weight:500}}
    input,select{{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:8px;color:#e6edf3;padding:10px 14px;font-size:.95rem;outline:none;transition:border-color .2s}}
    input:focus,select:focus{{border-color:#58a6ff;box-shadow:0 0 0 3px rgba(88,166,255,.15)}}
    select option{{background:#161b22}}
    .btn{{background:#238636;color:#fff;border:none;border-radius:8px;padding:10px 22px;font-size:.95rem;font-weight:600;cursor:pointer;white-space:nowrap;transition:background .2s}}
    .btn:hover{{background:#2ea043}}
    .chips{{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}}
    .chip{{background:#21262d;border:1px solid #30363d;border-radius:20px;padding:4px 14px;font-size:.8rem;cursor:pointer;color:#8b949e;transition:all .2s}}
    .chip:hover,.chip.active{{background:#58a6ff;border-color:#58a6ff;color:#0d1117;font-weight:600}}
    .chart-card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 16px;overflow:hidden}}
    .error-box{{background:#2d1b1b;border:1px solid #f85149;border-radius:8px;padding:16px 20px;color:#f85149}}
  </style>
</head>
<body>
<header>
  <div class="logo">📈 StockChart</div>
  <div class="subtitle">Live stock charts — US &amp; NSE markets</div>
</header>
<main>
  <div class="panel">
    <div class="panel-title">Search</div>
    <form method="POST" action="/">
      <div>
        <label for="ticker">Ticker Symbol</label>
        <input id="ticker" name="ticker" type="text" value="{ticker}"
               placeholder="e.g. AAPL, GOOGL, TCS.NS" required/>
      </div>
      <div>
        <label for="period">Time Range</label>
        <select id="period" name="period">{period_opts}</select>
      </div>
      <div>
        <label for="chart_type">Chart Type</label>
        <select id="chart_type" name="chart_type">
          <option value="candlestick" {ct_candle}>🕯 Candlestick</option>
          <option value="line" {ct_line}>📉 Line</option>
        </select>
      </div>
      <button type="submit" class="btn">Search →</button>
    </form>
    <div class="chips">{chips}</div>
  </div>
  <div class="chart-card">{content}</div>
</main>
<script>
  function setTicker(s){{document.getElementById('ticker').value=s;document.querySelector('form').submit()}}
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


@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    return (f"<pre style='background:#1a1a2e;color:#ff6b6b;padding:24px'>"
            f"<b>500 — Internal Server Error</b>\n\n{tb}</pre>"), 500


if __name__ == "__main__":
    app.run(debug=True)
