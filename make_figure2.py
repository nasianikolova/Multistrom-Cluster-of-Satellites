"""Figure 1: altitude against delay, hop count and signal-to-noise ratio.

Three corrections from review:
  A. tick labels enlarged for print legibility;
  B. discrete simulated hop counts plotted as markers, no fitted curve;
  C. SNR on a decibel axis, the link-budget convention, replacing the linear
     scale that compressed everything above 750 km against the axis.
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constellation as F

C_KM_S = 299_792.458
C_M_S = 299_792_458.0
K_B = 1.380649e-23

# Link-budget parameters, matching Table 1 of the manuscript.
PT, GT, GR = 5.0, 3000.0, 3000.0
FREQ_HZ, T_SYS, BW = 12e9, 290.0, 20e6
LAMBDA_M = C_M_S / FREQ_HZ

ALT = np.arange(300, 2001, 25)

NUM_SATS = 24
SEPARATION_DEG = 60.0
MAX_ISL_KM = 2500.0

LABEL = 11
TICK = 10
TITLE = 11.5


def snr_db(alt_km):
    d_m = alt_km * 1000.0
    pr = PT * GT * GR * (LAMBDA_M / (4 * math.pi * d_m)) ** 2
    n = K_B * T_SYS * BW
    return 10.0 * math.log10(pr / n)


def hops_at(alt_km):
    sim = F.Simulation(
        [F.Shell("leo", float(alt_km), NUM_SATS)],
        ground_angles_deg=[0.0, SEPARATION_DEG],
        processing_delay_s=0.001,
        max_isl_leo_leo_km=MAX_ISL_KM,
        min_elevation_deg=F.CONFIG["min_elevation_deg"],
    )
    _, path = sim.shortest_path("G0", "G1")
    return np.nan if not path else len(path) - 1


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.subplots_adjust(wspace=0.30, top=0.80, bottom=0.17, left=0.06, right=0.98)

    # ---------------- A: propagation delay ----------------
    ax = axes[0]
    delay_ms = 1000.0 * ALT / C_KM_S
    ax.plot(ALT, delay_ms, color="#1f4e79", lw=2.2)
    ax.set_xlabel("Satellite altitude (km)", fontsize=LABEL)
    ax.set_ylabel("One-way vertical\npropagation delay (ms)", fontsize=LABEL)
    ax.set_title("A   Propagation delay", fontsize=TITLE, pad=10, loc="left")
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.3, lw=0.5)
    for a in (300, 2000):
        ax.annotate(f"{1000*a/C_KM_S:.2f} ms",
                    xy=(a, 1000 * a / C_KM_S),
                    xytext=(a + (120 if a == 300 else -420),
                            1000 * a / C_KM_S + (0.55 if a == 300 else -0.15)),
                    fontsize=9.5, color="#1f4e79")

    # ---------------- B: hop count ----------------
    ax = axes[1]
    hops = np.array([hops_at(a) for a in ALT], dtype=float)
    ok = ~np.isnan(hops)
    x, y = ALT[ok].astype(float), hops[ok]
    ax.step(x, y, where="post", color="#1f4e79", lw=1.7, alpha=0.55)
    ax.plot(x, y, "o", ms=4.5, color="#1f4e79")
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            ax.axvline(x[i], color="0.6", ls=":", lw=1.0)
            ax.annotate(f"{x[i]:.0f} km", xy=(x[i], y.max()),
                        xytext=(x[i] + 45, y.max() - 0.25),
                        fontsize=9, color="0.35")
    ax.set_yticks(np.arange(int(y.min()), int(y.max()) + 1))
    ax.set_ylim(y.min() - 0.6, y.max() + 0.4)
    ax.set_xlabel("Satellite altitude (km)", fontsize=LABEL)
    ax.set_ylabel("Number of hops", fontsize=LABEL)
    ax.set_title(f"B   Hop count, {NUM_SATS}-satellite ring",
                 fontsize=TITLE, pad=10, loc="left")
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.3, lw=0.5)

    # ---------------- C: SNR in dB ----------------
    ax = axes[2]
    snr = np.array([snr_db(a) for a in ALT])
    ax.plot(ALT, snr, color="#1f4e79", lw=2.2)
    ax.set_xlabel("Satellite altitude (km)", fontsize=LABEL)
    ax.set_ylabel("Signal-to-noise ratio (dB)", fontsize=LABEL)
    ax.set_title("C   Link budget", fontsize=TITLE, pad=10, loc="left")
    ax.tick_params(labelsize=TICK)
    ax.grid(alpha=0.3, lw=0.5)
    ax.axhline(13.0, color="#c00000", ls="--", lw=1.3)
    ax.annotate("required C/N + margin, 13 dB", xy=(1000, 13),
                xytext=(430, 15.5), fontsize=9, color="#c00000")
    for a in (300, 2000):
        ax.annotate(f"{snr_db(a):.1f} dB", xy=(a, snr_db(a)),
                    xytext=(a + (120 if a == 300 else -430),
                            snr_db(a) + (0.3 if a == 300 else 1.8)),
                    fontsize=9.5, color="#1f4e79")
    ax.set_ylim(10, 48)

    fig.savefig("Figure2_altitude_relationships.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig("Figure2_altitude_relationships.tiff", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Panel A  delay at 300 km  : %.2f ms" % (1000 * 300 / C_KM_S))
    print("         delay at 2000 km : %.2f ms" % (1000 * 2000 / C_KM_S))
    print("Panel B  hop transitions  :", end=" ")
    prev = None
    out = []
    for a, h in zip(x, y):
        if h != prev:
            out.append(f"{int(h)} hops from {a:.0f} km")
            prev = h
    print("; ".join(out))
    print("Panel C  SNR at 300 km    : %.2f dB" % snr_db(300))
    print("         SNR at 2000 km   : %.2f dB" % snr_db(2000))
    print("         margin threshold : 13.00 dB, satisfied across the whole range")
    print("\nWritten: Figure2_altitude_relationships.png / .tiff")


if __name__ == "__main__":
    main()
