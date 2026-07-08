"""
Data ingestion: ticker universe, price download, volume, VADER sentiment.
Extracted from pipeline_course.py — all caching logic preserved.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Optional

# ── Default cache directory (same folder as the calling script) ────────────
_DEFAULT_CACHE = os.path.dirname(os.path.abspath(__file__))

# ── Hardcoded S&P 500 + NASDAQ 100 ticker list ────────────────────────────
_SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AVGO","ORCL","AMD","QCOM","TXN","AMAT","MU",
    "LRCX","KLAC","ADI","MCHP","CDNS","SNPS","FTNT","PANW","CRWD","ZS",
    "DDOG","NET","SNOW","PLTR","UBER","TTD","VEEV","PAYC","ANSS","TER",
    "KEYS","TRMB","FSLR","ENPH","HPE","HPQ","DELL","WDC","STX","NTAP",
    "CDW","ZBRA","FFIV","AKAM","EPAM","CTSH","IT","ACN","IBM","LDOS",
    "GOOGL","GOOG","META","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR",
    "WBD","PARA","FOX","FOXA","SIRI","SPOT","EA","TTWO","RBLX","MTCH",
    "AMZN","TSLA","MCD","NKE","SBUX","HD","LOW","TJX","BKNG","MAR",
    "HLT","YUM","CMG","RCL","CCL","NCLH","GM","F","APTV","LVS","WYNN",
    "MGM","CZR","DKNG","ORLY","AZO","KMX","AN","BBY","ROST","ETSY",
    "EBAY","ABNB","LYFT","DASH","CPRT","XPO","ODFL","PDD","JD","MELI",
    "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","KHC","GIS","CPB",
    "HRL","MKC","CL","EL","CHD","CLX","ENR","UL","STZ","MNST","CELH",
    "XOM","CVX","COP","EOG","SLB","MPC","VLO","PSX","DVN","FANG","APA",
    "HAL","BKR","OXY","KMI","WMB","OKE","ET","EPD","MPLX","ENB","TRP",
    "BRK-B","JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","USB",
    "PNC","TFC","COF","SYF","AIG","MET","PRU","AFL","ALL","PGR","TRV",
    "CB","HIG","AON","SPGI","MCO","ICE","CME","CBOE","NDAQ","FIS","FISV",
    "GPN","ADP","PAYX","BR","MSCI","VRSK","UNH","JNJ","LLY","ABBV","MRK",
    "ABT","TMO","DHR","BMY","AMGN","GILD","VRTX","REGN","ISRG","SYK",
    "MDT","BSX","EW","BDX","IDXX","IQV","CRL","ILMN","BIIB","MRNA","PFE",
    "CI","CVS","HUM","CNC","MOH","ELV","MCK","CAH","GE","HON","RTX","BA",
    "LMT","NOC","GD","TDG","CAT","DE","EMR","ETN","PH","ROK","SWK","IR",
    "XYL","AME","FTV","OTIS","CARR","TT","JCI","FAST","UPS","FDX","DAL",
    "UAL","AAL","LUV","CSX","UNP","NSC","URI","RSG","WM","CTAS","PCAR",
    "LIN","APD","ECL","SHW","PPG","RPM","DD","DOW","LYB","IFF","CE","EMN",
    "ALB","NUE","STLD","RS","FCX","NEM","GOLD","AEM","WPM","PLD","AMT",
    "EQIX","CCI","SPG","PSA","EQR","MAA","AVB","ESS","IRM","WELL","VTR",
    "ARE","BXP","SLG","KIM","REG","O","NNN","VICI","GLPI","SBAC","DLR",
    "NEE","DUK","SO","AEP","EXC","SRE","D","PCG","PEG","XEL","ETR","FE",
    "PPL","AES","CNP","CMS","NI","WEC","LNT","EVRG","AWK",
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","IEF","HYG","LQD",
    "VXX","EEM","EFA","VEA","VWO",
]


def get_tickers(use_wikipedia: bool = True) -> List[str]:
    """Return the S&P 500 ticker universe. Falls back to hardcoded list."""
    if use_wikipedia:
        try:
            tables  = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                flavor="lxml")
            tickers = (tables[0]["Symbol"]
                       .str.replace(".", "-", regex=False)
                       .tolist())
            print(f"Wikipedia: {len(tickers)} tickers.")
            return tickers
        except Exception as e:
            print(f"Wikipedia failed ({e}). Using hardcoded list.")
    return _SP500_TICKERS


def download_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Download (or load cached) adjusted close prices."""
    cache_dir = cache_dir or _DEFAULT_CACHE
    end       = end or datetime.today().strftime("%Y-%m-%d")
    cache     = os.path.join(cache_dir, "close_cache_full.parquet")

    if os.path.exists(cache):
        print("Loading cached price data ...", flush=True)
        close = pd.read_parquet(cache)
        print(f"  {close.shape[1]} tickers x {close.shape[0]} days.")
        return close

    print(f"Downloading {len(tickers)} tickers ...", flush=True)
    frames = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    for bi, batch in enumerate(batches):
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            ok = c.dropna(axis=1, thresh=int(0.7*len(c))).columns.tolist()
            frames.append(c[ok])
        except Exception as e:
            print(f"  batch {bi+1} failed: {e}", flush=True)
        print(f"  batch {bi+1}/{len(batches)} done", flush=True)

    if not frames:
        raise RuntimeError("No price data downloaded.")

    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    close = close.dropna(axis=1, thresh=int(0.7*len(close))).ffill().dropna()
    close.to_parquet(cache)
    print(f"  Saved {close.shape[1]} tickers x {close.shape[0]} days.")
    return close


def download_volume(
    tickers: List[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
) -> Optional[pd.DataFrame]:
    """Download (or load cached) daily volume."""
    cache_dir = cache_dir or _DEFAULT_CACHE
    end       = end or datetime.today().strftime("%Y-%m-%d")
    cache     = os.path.join(cache_dir, "vol_cache_full.parquet")

    if os.path.exists(cache):
        print("Loading cached volume data ...", flush=True)
        vol = pd.read_parquet(cache)
        print(f"  {vol.shape[1]} tickers x {vol.shape[0]} days.")
        return vol

    print(f"Downloading volume for {len(tickers)} tickers ...", flush=True)
    frames = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    for bi, batch in enumerate(batches):
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            v = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw
            ok = v.dropna(axis=1, thresh=int(0.7*len(v))).columns.tolist()
            frames.append(v[ok])
        except Exception as e:
            print(f"  batch {bi+1} failed: {e}", flush=True)
        print(f"  batch {bi+1}/{len(batches)} done", flush=True)

    if not frames:
        print("No volume data downloaded.")
        return None

    vol = pd.concat(frames, axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()]
    vol = vol.dropna(axis=1, thresh=int(0.7*len(vol))).ffill().fillna(0)
    vol.to_parquet(cache)
    print(f"  Saved {vol.shape[1]} tickers x {vol.shape[0]} days.")
    return vol


def fetch_sentiment(
    tickers: List[str],
    close_index: pd.Index,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch and cache VADER NLP sentiment scores from Yahoo Finance news."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    cache_dir = cache_dir or _DEFAULT_CACHE
    cache     = os.path.join(cache_dir, "sentiment_cache.parquet")

    if os.path.exists(cache):
        print("Loading cached sentiment scores ...", flush=True)
        return pd.read_parquet(cache).reindex(close_index).ffill(limit=3).fillna(0.0)

    print(f"Fetching news for {len(tickers)} tickers (VADER) ...", flush=True)
    analyzer = SentimentIntensityAnalyzer()
    records  = []

    for i, t in enumerate(tickers):
        try:
            news_items = yf.Ticker(t).news or []
            for item in news_items:
                content = item.get("content", item)
                raw_ts  = (content.get("pubDate")
                           or content.get("providerPublishTime")
                           or item.get("providerPublishTime"))
                if raw_ts is None:
                    continue
                date  = (pd.Timestamp(raw_ts).normalize()
                         if isinstance(raw_ts, str)
                         else pd.Timestamp(int(raw_ts), unit="s").normalize())
                title = (content.get("title") or content.get("headline")
                         or item.get("title", ""))
                if not title:
                    continue
                score = analyzer.polarity_scores(str(title))["compound"]
                records.append({"date": date, "ticker": t, "score": score})
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  Sentiment: {i+1}/{len(tickers)} ...", flush=True)

    if not records:
        sent = pd.DataFrame(0.0, index=close_index, columns=tickers)
        sent.to_parquet(cache)
        return sent

    df_news    = pd.DataFrame(records)
    sent_pivot = (df_news.groupby(["date", "ticker"])["score"]
                  .mean().unstack(fill_value=np.nan))
    sent = (sent_pivot.reindex(close_index)
            .reindex(columns=tickers, fill_value=np.nan)
            .ffill(limit=3).fillna(0.0))
    sent.to_parquet(cache)
    return sent
