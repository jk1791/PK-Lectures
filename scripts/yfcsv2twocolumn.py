import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="LPPL model fit")
parser.add_argument("filename",help="data file")
args = parser.parse_args()

input_file = args.filename
output_file = input_file[:-4] + "_converted.dat"

# Wczytanie danych bez nagłówka
df = pd.read_csv(input_file, header=None)

# Twoja kolejność kolumn:
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Funkcja konwersji daty na postać: RRRR + (dzien_roku / liczba_dni_w_roku)
def date_to_year_fraction(date_str):
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year = d.year
    
    # dzień roku (1–365 lub 366)
    day_of_year = d.timetuple().tm_yday
    
    # liczba dni w roku (365 lub 366)
    days_in_year = 366 if is_leap_year(year) else 365
    
    return year + (day_of_year - 1) / days_in_year

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Przeliczanie kolumn
df["DateFloat"] = df["Date"].apply(date_to_year_fraction)
df["CloseFloat"] = df["Close"].astype(float)

# Formatowanie wyjściowe
df["DateFmt"] = df["DateFloat"].map(lambda x: f"{x:10.5f}")
df["CloseFmt"] = df["CloseFloat"].map(lambda x: f"{x:10.2f}")

# Zapis dwóch kolumn
with open(output_file, "w") as f:
    for d, c in zip(df["DateFmt"], df["CloseFmt"]):
        f.write(f"{d} {c}\n")

print("Gotowe:", output_file)

