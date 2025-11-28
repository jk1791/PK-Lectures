#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Rysuje wykres lin-log z pliku z kolumnami oddzielonymi spacjami."
    )
    parser.add_argument("filename", help="plik z danymi (kolumny oddzielone spacjami)")
    parser.add_argument(
        "-c", "--column",
        type=int,
        default=2,
        help="numer kolumny (1-based) użytej jako oś Y (domyślnie 2)"
    )
    args = parser.parse_args()

    # Wczytanie danych (bez nagłówka; jeśli masz nagłówek, dodaj skiprows=1)
    data = np.loadtxt(args.filename)

    if args.column < 2:
        raise ValueError("Kolumna dla osi Y musi mieć numer co najmniej 2 (1. kolumna to czas).")

    if args.column > data.shape[1]:
        raise ValueError(
            f"Plik ma tylko {data.shape[1]} kolumn, nie mogę użyć kolumny nr {args.column}."
        )

    t = data[:, 0]                 # 1. kolumna – czas
    y = data[:, args.column - 1]   # kolumna wybrana parametrem

    # Wykres w skali lin-log (X liniowo, Y logarytmicznie)
    plt.figure()
    plt.semilogy(t, y)

    plt.xlabel("czas")
    plt.ylabel(f"kolumna {args.column} (skala log Y)")
    plt.title(f"{args.filename} — kolumna {args.column} vs czas (lin-log)")
    plt.grid(True, which="both")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

