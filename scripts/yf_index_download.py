import yfinance as yf
import argparse

parser = argparse.ArgumentParser(description="LPPL model fit")
parser.add_argument("filename",help="data file")
parser.add_argument("--asset",type=str,help="asset name")
args = parser.parse_args()

asset_name = yf.download(args.asset,period="max",interval="1d")
asset_name.to_csv(args.filename)

