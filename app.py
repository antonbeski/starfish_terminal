import traceback
from flask import Flask, render_template, request
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd

app = Flask(__name__)

POPULAR_STOCKS = [
    {"symbol": "AAPL",        "name": "Apple"},
    {"symbol": "GOOGL",       "name": "Google"},
    {"symbol": "MSFT",        "name": "Microsoft"},
    {"symbol": "TSLA",        "name": "Tesla"},
    {"symbol": "AMZN",        "name": "Amazon"},
    {"symbol": "NVDA",        "name": "NVIDIA"},
    {"symbol": "TCS.NS",      "name": "TCS (NSE)"},
    {"symbol": "RELIANCE.NS", "name": "Reliance (NSE)"},
]

PERIODS = [
    {"value": "1mo", "label": "1 Month"},
    {"value": "3mo", "label": "3 Months"},
    {"value": "6mo", "label": "6 Months"},
    {"value": "1y",  "label": "1 Year"},
    {"value": "2y",  "label": "2 Years"},
    {"value": "5y",  "label": "5 Years"},
]

VALID_PERIODS = {p["value"] for p in PERIODS}


def flatten_columns(df):
    """Flatten MultiIndex columns that newer yfinance versions return."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_chart(ticker, period, chart_type):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            actions=False,
        )
    except Exception as e:
        return None, f"Download failed: {e}"

    if data is None or data.empty:
        return None, (
            f"No data returned for '{ticker}'. "
            "Check the symbol — use '.NS' suffix for NSE stocks (e.g. TCS.NS)."
        )

    data = flatten_columns(data)

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(data.columns)
    if missing:
        return None, f"Unexpected data format — missing columns: {missing}."

    data = data.dropna(subset=["Close"])
    if data.empty:
        return None, f"All data rows for '{ticker}' were empty after cleaning."

    try:
        info = yf.Ticker(ticker).fast_info
        name = getattr(info, "company_name", None) or ticker
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
            increasing_line_color="#26a641",
            decreasing_line_color="#f85149",
        )
    else:
        trace = go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=ticker,
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        )

    fig = go.Figure(data=trace)
    fig.update_layout(
        title=dict(text=f"{name} ({ticker.upper()})", font=dict(size=20, color="#e6edf3")),
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#21262d", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#21262d"),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    graph_html = pyo.plot(fig, output_type="div", include_plotlyjs=False)
    return graph_html, None


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

    return render_template(
        "index.html",
        graph=graph_html,
        error=error,
        ticker=ticker,
        period=period,
        chart_type=chart_type,
        popular_stocks=POPULAR_STOCKS,
        periods=PERIODS,
    )


@app.errorhandler(500)
def internal_error(e):
    tb = traceback.format_exc()
    return (
        f"<pre style='background:#1a1a2e;color:#ff6b6b;padding:24px'>"
        f"<b>500 — Internal Server Error</b>\n\n{tb}</pre>"
    ), 500


@app.errorhandler(404)
def not_found(e):
    return "<h2>404 — Page not found</h2>", 404


if __name__ == "__main__":
    app.run(debug=True)
