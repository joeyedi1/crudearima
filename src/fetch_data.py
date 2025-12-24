import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
TICKERS = {
    'NQ_GOOGLE': 'NQW00', # Note: yfinance might not have Google's specific tickers like NQW00, using NQ=F as primary for Yahoo
    'NQ_YAHOO': 'NQ=F',
    'VIX': '^VIX',
    'TNX': '^TNX', # Yield / 10
    'DXY': 'DX-Y.NYB',
    'BRENT': 'BZ=F',
    'GOLD': 'GC=F',
    'BTC': 'BTC-USD',
    'VXN': '^VXN' # Nasdaq Volatility
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_data():
    """Fetches all required data and returns a dictionary of DataFrames/Values."""
    logging.info("Fetching market data...")
    
    # 1. Fetch NQ Data (Intraday for 4H construction)
    # yfinance 60m is the granular available for free for last 730 days.
    # We will use 60m to construct 4H.
    nq_hourly = yf.download(TICKERS['NQ_YAHOO'], period="50d", interval="60m", progress=False, group_by='column', auto_adjust=True)
    
    # Handle auto_adjust=True return format (removes 'Adj Close' level if just one ticker, but checking just in case)
    if isinstance(nq_hourly.columns, pd.MultiIndex):
        nq_hourly = nq_hourly.xs(TICKERS['NQ_YAHOO'], axis=1, level=1)
    
    # 2. Fetch Daily Cross Assets
    cross_tickers = [TICKERS['VIX'], TICKERS['TNX'], TICKERS['DXY'], TICKERS['BRENT'], TICKERS['GOLD'], TICKERS['BTC']]
    cross_data = yf.download(cross_tickers, period="5d", progress=False, group_by='column', auto_adjust=True)
    
    return nq_hourly, cross_data

def process_nq_4h(df_hourly):
    """Resamples hourly NQ data to 4H (Globex aligned)."""
    # 02/06/10/14/18/22 ET alignment
    # Ensure index is tz-aware ET
    if df_hourly.index.tz is None:
         df_hourly.index = df_hourly.index.tz_localize('UTC') # yf returns UTC usually
    df_hourly = df_hourly.tz_convert('America/New_York')

    # Custom resample to align with 18:00 ET start of day (Globex)
    # We want 6 bars per day. 18, 22, 02, 06, 10, 14.
    # offset='18h' means bins start at 18:00.
    
    df_4h = (df_hourly.resample('4h', origin='start_day', offset='18h')
             .agg({'Open':'first','High':'max','Low':'min','Close':'last'})
             .dropna())
    
    # Calculate SMA(20) on 4H
    df_4h['SMA20'] = df_4h['Close'].rolling(window=20).mean()
    
    return df_4h

def calculate_atr_daily(df_4h):
    """Calculates Daily ATR(5) from 4H data using Globex Daily definition."""
    # Globex Day: 18:00 ET previous day -> 17:00 ET current day
    
    df = df_4h.copy()
    
    # Define session date: if hour >= 18, it belongs to NEXT day's session logic or same?
    # User formula: date(Datetime - 18h).
    df['SessionDate'] = (df.index - pd.Timedelta(hours=18)).date
    
    daily_agg = df.groupby('SessionDate').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'SMA20': 'last' # just to keep it
    })
    
    # ATR(5) Calculation (Wilder's Smoothing)
    # TR = Max(H-L, |H-Cp|, |L-Cp|)
    
    high = daily_agg['High']
    low = daily_agg['Low']
    close = daily_agg['Close']
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Wilder's Smoothing: alpha = 1/n
    # Pandas ewm com = n - 1 (for alpha=1/(1+com) -> alpha=1/n => 1+com=n => com=n-1) ?
    # No, Wilder's RMA is often defined as separate. 
    # Pandas ewm(alpha=1/n, adjust=False) matches Wilder's if initialized correctly.
    # Standard ATR usually uses RMA (Running Moving Average).
    # RMA(5) = EWM(alpha=1/5, adjust=False)
    
    daily_agg['ATR5'] = tr.ewm(alpha=1/5, adjust=False).mean()
    
    return daily_agg

def get_latest_cross_assets(cross_data):
    """Extracts latest prices and changes for cross assets."""
    # Structure of cross_data columns: (Price, Ticker)
    
    metrics = {}
    
    # Helper to get last valid value
    def get_last(ticker):
        try:
            return cross_data.xs(ticker, axis=1, level=1)['Close'].dropna().iloc[-1]
        except:
            return 0.0

    metrics['TNX'] = get_last(TICKERS['TNX'])
    metrics['VIX'] = get_last(TICKERS['VIX'])
    metrics['DXY'] = get_last(TICKERS['DXY'])
    metrics['Brent'] = get_last(TICKERS['BRENT'])
    metrics['Gold'] = get_last(TICKERS['GOLD'])
    metrics['BTC'] = get_last(TICKERS['BTC'])
    
    return metrics

def main():
    try:
        logging.info("Starting Data Fetch...")
        nq_hourly, cross_data = fetch_data()
        
        nq_4h = process_nq_4h(nq_hourly)
        nq_daily = calculate_atr_daily(nq_4h)
        cross_metrics = get_latest_cross_assets(cross_data)
        
        # Save processed data for report generator
        nq_4h.to_pickle(DATA_DIR / "nq_4h.pkl")
        nq_daily.to_pickle(DATA_DIR / "nq_daily.pkl")
        pd.DataFrame([cross_metrics]).to_pickle(DATA_DIR / "cross_metrics.pkl")
        
        logging.info(f"Data saved to {DATA_DIR}")
        
    except Exception as e:
        logging.critical(f"Data Fetch Failed: {e}")
        raise

if __name__ == "__main__":
    main()
