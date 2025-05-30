import yfinance as yf

def check_symbol_data(symbol):
    try:
        df = yf.download(symbol, start='1900-01-01')
        return not df.empty
    except Exception:
        return False
