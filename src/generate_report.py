import pandas as pd
import os
import logging
from pathlib import Path
from datetime import datetime
import openai
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESS_DIR = BASE_DIR / "data/processed"
TEMPLATE_PATH = BASE_DIR / "notebooks/note"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Loads processed data from pickles."""
    try:
        nq_4h = pd.read_pickle(PROCESS_DIR / "nq_4h.pkl")
        nq_daily = pd.read_pickle(PROCESS_DIR / "nq_daily.pkl")
        cross_metrics = pd.read_pickle(PROCESS_DIR / "cross_metrics.pkl").iloc[0].to_dict()
        return nq_4h, nq_daily, cross_metrics
    except FileNotFoundError as e:
        logging.error(f"Data file not found. Run fetch_data.py first. Error: {e}")
        raise

def format_data_block(nq_4h, nq_daily, cross_metrics):
    """Formats the data into a string block to append to the prompt."""
    
    last_4h = nq_4h.iloc[-1]
    last_daily = nq_daily.iloc[-1]
    
    # Calculate Deviation (Simulated as 0 since we only used one source `NQ=F`)
    current_price = last_4h['Close']
    deviation_str = "0.00% (Single Source Mode)"
    
    # ATR(5)
    atr5 = last_daily.get('ATR5', 0.0)
    
    # SMA(20) on 4H
    sma20 = last_4h.get('SMA20', 0.0)
    
    # OHLC Data String (Last 6 4H bars for context if needed, or just recent)
    ohlc_str = nq_4h.tail(6).to_string()

    data_block = f"""
---
# [SYSTEM: AUTOMATICALLY INJECTED LIVE DATA]
# Data Timestamp (ET): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. NQ Price & Indicators
- Current Price: {current_price:.2f}
- Deviation (Google vs Yahoo): {deviation_str}
- 4H SMA(20): {sma20:.2f}
- Daily ATR(5, Wilder): {atr5:.2f}

## 2. Cross Asset Metrics
- TNX (10Y Yield): {cross_metrics.get('TNX', 0):.2f}%
- VIX: {cross_metrics.get('VIX', 0):.2f}
- DXY: {cross_metrics.get('DXY', 0):.2f}
- Brent Crude: {cross_metrics.get('Brent', 0):.2f}
- Gold: {cross_metrics.get('Gold', 0):.2f}
- BTC: {cross_metrics.get('BTC', 0):.2f}

## 3. Recent 4H OHLC (Last 6 bars)
{ohlc_str}
---
"""
    return data_block

def generate_prompt(template_content, data_block):
    """Combines template and data block."""
    return template_content + "\\n" + data_block

def call_llm(prompt):
    """Calls OpenAI or Gemini API if key exists."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if gemini_key:
        try:
            logging.info("Using Gemini API...")
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API Call Failed: {e}")
            return None

    elif openai_key:
        try:
            logging.info("Using OpenAI API...")
            client = openai.Client(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst. Generate the report exactly following the user's template."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API Call Failed: {e}")
            return None
    
    else:
        logging.warning("No API Key (OPENAI_API_KEY or GEMINI_API_KEY) found. Skipping LLM generation.")
        return None

def main():
    try:
        # Load Data
        nq_4h, nq_daily, cross_metrics = load_data()
        
        # Read Template
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_content = f.read()
            
        # Format Data
        data_block = format_data_block(nq_4h, nq_daily, cross_metrics)
        full_prompt = generate_prompt(template_content, data_block)
        
        # Save Prompt (Always)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_file = OUTPUT_DIR / f"prompt_{timestamp}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(full_prompt)
        logging.info(f"Prompt saved to {prompt_file}")
        
        # Call LLM (Optional)
        report_content = call_llm(full_prompt)
        
        if report_content:
            report_file = OUTPUT_DIR / f"report_{timestamp}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            logging.info(f"Report generated and saved to {report_file}")
        else:
            logging.info("Report generation skipped (No API Key). Use the prompt file with ChatGPT manually.")
            
    except Exception as e:
        logging.critical(f"Report Generation Failed: {e}")

if __name__ == "__main__":
    main()
