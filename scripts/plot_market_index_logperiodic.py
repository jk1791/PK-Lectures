#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- opcjonalnie: dopasowanie ---
try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("Uwaga: scipy nie jest dostępne – będzie tylko ręczne strojenie parametrów.")

# --- model: f(t) = C |tc - t|^{-dC} [1 + 2 B cos( 2π log|tc - t| / log λ + φ )] ---
def lppl(t, C, B, tc, phi, dC, lam):
    t = np.asarray(t)
    dt = np.abs(tc - t)
    # zabezpieczenie przed log(0)
    dt = np.where(dt <= 0, np.nan, dt)
    return C * dt ** (-dC) * (1.0 + 2.0 * B * np.cos(2.0 * np.pi * np.log(dt) / np.log(lam) + phi))


def main():

    parser = argparse.ArgumentParser(description="LPPL model fit")
    parser.add_argument("filename", help="plik z danymi (kolumny oddzielone spacjami)")
    parser.add_argument("--column",type=int,default=2,help="numer kolumny (1-based) użytej jako oś Y (domyślnie 2)")
    parser.add_argument("--no-fit",action="store_true",help="nie próbuj automatycznie dopasowywać parametrów (tylko slidery)")
    parser.add_argument("--asset",type=str,default="value",help="asset name")
    args = parser.parse_args()

    # --- wczytanie danych ---
    data = np.loadtxt(args.filename)

    if args.column < 2:
        raise ValueError("Kolumna dla osi Y musi mieć numer co najmniej 2 (1. kolumna to czas).")

    if args.column > data.shape[1]:
        raise ValueError(
            f"Plik ma tylko {data.shape[1]} kolumn, nie mogę użyć kolumny nr {args.column}."
        )

    t = data[:, 0]
    y = data[:, args.column - 1]

    # --- wartości startowe parametrów modelu ---
    t_min, t_max = np.min(t), np.max(t)
    y_med = np.median(y[~np.isnan(y) & (y > 0)]) if np.any(y > 0) else 1.0

    C0 = y_med
    B0 = 0.1
    tc0 = t_max + 0.1 * (t_max - t_min)   # trochę za końcem danych
    phi0 = 0.0
    dC0 = 0.5
    lam0 = 2.0

    p0 = np.array([C0, B0, tc0, phi0, dC0, lam0])

    # --- opcjonalne automatyczne dopasowanie ---
    if HAVE_SCIPY and not args.no_fit:
        try:
            # sensowne ograniczenia:
            # C > 0, |B| < 1, tc powyżej max(t), dC > 0, lam > 1
            bounds_lower = [0.0,   -1.0,   t_max,   -4*np.pi,  0.0,  1.01]
            bounds_upper = [np.inf, 1.0,   t_max*1.1,  4*np.pi,  5.0, 10.0]

            popt, pcov = curve_fit(
                lppl, t, y,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000
            )
            C0, B0, tc0, phi0, dC0, lam0 = popt
            print("Dopasowane parametry startowe:")
            print(f"C   = {C0}")
            print(f"B   = {B0}")
            print(f"tc  = {tc0}")
            print(f"phi = {phi0}")
            print(f"dC  = {dC0}")
            print(f"lam = {lam0}")
        except Exception as e:
            print("Nie udało się dopasować modelu, używam wartości domyślnych.")
            print("Powód:", e)

    # --- rysowanie ---
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.10, bottom=0.35)  # miejsce na slidery

    # dane w skali lin-log (X liniowo, Y logarytmicznie)
    data_line = ax.semilogy(t, y, ".", label="dane")[0]

    # linia modelu
    y_model = lppl(t, C0, B0, tc0, phi0, dC0, lam0)
    model_line, = ax.semilogy(t, y_model, "-", label="model")

    ax.set_xlabel("czas")
    ax.set_ylabel(f"{args.asset} [USD]")
    ax.set_title(f"{args.filename} — kolumna {args.column} vs czas (lin-log)")
    ax.grid(True, which="both")
    ax.legend()

    # --- slidery dla parametrów ---
    # oś sliderów: [left, bottom, width, height]
    axcolor = "lightgoldenrodyellow"
    ax_C   = plt.axes([0.10, 0.27, 0.80, 0.03], facecolor=axcolor)
    ax_B   = plt.axes([0.10, 0.23, 0.80, 0.03], facecolor=axcolor)
    ax_tc  = plt.axes([0.10, 0.19, 0.80, 0.03], facecolor=axcolor)
    ax_phi = plt.axes([0.10, 0.15, 0.80, 0.03], facecolor=axcolor)
    ax_dC  = plt.axes([0.10, 0.11, 0.80, 0.03], facecolor=axcolor)
    ax_lam = plt.axes([0.10, 0.07, 0.80, 0.03], facecolor=axcolor)

    # zakresy sliderów — możesz dostosować pod swoje dane
    sC   = Slider(ax_C,   "C",   0.1 * C0 if C0 != 0 else 1e-3, 10 * (C0 if C0 != 0 else 1.0), valinit=C0)
    sB   = Slider(ax_B,   "B",  -1.0, 1.0,      valinit=B0)
    stc  = Slider(ax_tc,  "tc", t_min, t_max*1.02, valinit=tc0, valstep=0.01)
    sphi = Slider(ax_phi, "phi", -4*np.pi, 4*np.pi, valinit=phi0)
    sdC  = Slider(ax_dC,  "dC",  0.0, 5.0, valinit=dC0)
    slam = Slider(ax_lam, "lam", 1.01, 10.0, valinit=lam0)

    # tekst z błędem dopasowania
    error_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top"
    )

    def update(val):
        C   = sC.val
        B   = sB.val
        tc  = stc.val
        phi = sphi.val
        dC  = sdC.val
        lam = slam.val

        y_mod = lppl(t, C, B, tc, phi, dC, lam)
        model_line.set_ydata(y_mod)

        # prosty błąd RMS (logarytmiczny dla dodatnich y)
        mask = (y > 0) & np.isfinite(y_mod) & (y_mod > 0)
        if np.any(mask):
            err = np.sqrt(np.mean((np.log(y[mask]) - np.log(y_mod[mask]))**2))
            error_text.set_text(f"RMS(log) = {err:.3e}")
        else:
            error_text.set_text("RMS(log) = ---")

        fig.canvas.draw_idle()

    # podpięcie callbacków
    sC.on_changed(update)
    sB.on_changed(update)
    stc.on_changed(update)
    sphi.on_changed(update)
    sdC.on_changed(update)
    slam.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()

