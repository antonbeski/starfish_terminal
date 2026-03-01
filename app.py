from flask import Flask, render_template, request
import yfinance as yf
import plotly.graph_objects as go
import plotly.offline as pyo
import pandas as pd

app = Flask(__name__)

POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "GOOGL", "name": "Google"},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "NVDA", "name": "NVIDIA"},
    {"symbol": "TCS.NS", "name": "TCS (NSE)"},
    {"symbol": "RELIANCE.NS", "name": "Reliance (NSE)"},
]

PERIODS = [
    {"value": "1mo", "label": "1 Month"},
    {"value": "3mo", "label": "3 Months"},
    {"value": "6mo", "label": "6 Months"},
    {"value": "1y", "label": "1 Year"},
    {"value": "2y", "label": "2 Years"},
    {"value": "5y", "label": "5 Years"},
]


def build_chart(ticker, period, chart_type):
    data = yf.download(ticker, period=period, interval="1d", progress=False)

    if data.empty:
        return None, f"No data found for ticker '{ticker}'. Please check the symbol."

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    stock_info = yf.Ticker(ticker)
    try:
        name = stock_info.info.get("shortName", ticker)
    except Exception:
        name = ticker

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
        yaxis_title="Price (USD)" if ".NS" not in ticker.upper() else "Price (INR)",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        xaxis=dict(
            gridcolor="#21262d",
            showgrid=True,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(gridcolor="#21262d", showgrid=True),
        hovermode="x unified",
        margin=dict(l=50, r=30, t=60, b=50),
    )

    graph_html = pyo.plot(fig, output_type="div", include_plotlyjs=False)
    return graph_html, None


@app.route("/", methods=["GET", "POST"])
def index():
    ticker = request.form.get("ticker", "AAPL").strip().upper()
    period = request.form.get("period", "6mo")
    chart_type = request.form.get("chart_type", "candlestick")

    # Validate period
    valid_periods = [p["value"] for p in PERIODS]
    if period not in valid_periods:
        period = "6mo"

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


if __name__ == "__main__":
    app.run(debug=True)
