import yfinance as yf 
import pandas as pd 
from tqdm import tqdm
from pathlib import Path 

from src.utils.config import (START_DATE,END_DATE,RAW_DATA_OUTPUT_PATH,RAW_BENCHMARK_OUTPUT_PATH,NIFTY50_TICKERS,BENCHMARK_TICKER)


def verify_directories():
    Path(RAW_DATA_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(RAW_BENCHMARK_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    

def fetch_stocks():
    data=[] 
    
    for ticker in tqdm(NIFTY50_TICKERS):
        df=yf.download(ticker,start=START_DATE,end=END_DATE,progress=False)
        
        if df.empty:
            continue
        
        df.reset_index(inplace=True)
        df["Ticker"]=ticker
        data.append(df)
    
    df=pd.concat(data,ignore_index=True)
    
    df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    df.sort_values(["Ticker", "Date"], inplace=True) 
    
    return df 

def fetch_benchmark():
    df=yf.download(BENCHMARK_TICKER,start=START_DATE,end=END_DATE,progress=False)
    
    df.reset_index(inplace=True)
    df.rename(columns={"Adj Close": "Adj_Close"},inplace=True)
    df.sort_values("Date",inplace=True)
    
    
    return df 

def main():
    verify_directories()
    
    stocks=fetch_stocks()
    stocks.to_parquet(RAW_DATA_OUTPUT_PATH,index=False)
    
    benchmark=fetch_benchmark()
    benchmark.to_parquet(RAW_BENCHMARK_OUTPUT_PATH,index=False)
    
    print("Data Saved Succesfully")
    
if __name__=="__main__":
    main()
    