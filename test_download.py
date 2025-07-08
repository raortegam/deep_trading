from functions.download_data import TradingDataDownloader
import numpy as np
import pandas as pd

# Test the download with debug information
try:
    print("Starting data download...")
    downloader = TradingDataDownloader("AAPL", period='1y')
    print("Downloader created successfully")
    
    # Try to download the data
    success = downloader.download_data()
    print(f"Download successful: {success}")
    
    if success and downloader.data is not None:
        print("\nDataFrame Info:")
        print(downloader.data.info())
        print("\nFirst 5 rows:")
        print(downloader.data.head())
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
