# 📈 StockChart — Live Stock Charts

A clean, dark-themed stock chart web app built with Flask, yfinance, and Plotly. Supports US and NSE (India) markets.

## Features

- 🕯 Candlestick & line charts
- 📅 Time ranges: 1M, 3M, 6M, 1Y, 2Y, 5Y
- 🇺🇸 US stocks (AAPL, GOOGL, TSLA, NVDA…)
- 🇮🇳 NSE stocks (TCS.NS, RELIANCE.NS…)
- ⚡ Quick-pick popular tickers
- 🌑 Dark GitHub-inspired theme

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/stock-chart.git
cd stock-chart

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy (from project root)
vercel

# Follow the prompts — select Python, confirm defaults
```

The `vercel.json` is already configured. Your app will be live at `https://your-project.vercel.app`.

## Project Structure

```
stock-chart/
├── app.py               # Flask app
├── templates/
│   └── index.html       # Jinja2 template
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel deployment config
├── .gitignore
└── README.md
```

## Example Ticker Symbols

| Market | Symbol |
|--------|--------|
| Apple (NASDAQ) | `AAPL` |
| Google (NASDAQ) | `GOOGL` |
| Tesla (NASDAQ) | `TSLA` |
| TCS (NSE) | `TCS.NS` |
| Reliance (NSE) | `RELIANCE.NS` |
| Infosys (NSE) | `INFY.NS` |
