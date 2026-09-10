"""Orbital and ground-relay baselines for the Sydney-Tokyo route.

Recommended Issue 3 asks how the multistrom geometry compares against orbital
or ground relay geometries. Four architectures are compared, all of them
in-space alternatives for the same route and the same endpoints:

    A  Bent-pipe LEO       ground - one LEO satellite - ground, no ISLs
    B  Ground relay        ground - satellite - intermediate ground station -
                           satellite - ground, still with no ISLs
    C  Single MEO relay    ground - one MEO satellite - ground
    D  Meshed LEO + ISLs   the architecture of this paper

Architecture B is the way a satellite network reaches beyond a single
satellite's footprint without inter-satellite links: it comes back down to a
relay station on the ground and goes up again. It is the direct alternative to
using ISLs, and it is the comparison the reviewer asked for.

Candidate ground relay sites are real locations on the Sydney-Tokyo corridor.
"""

import math
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figure5 as RT

R = RT.R_EARTH_KM
C = RT.C_KM_S
PROC_MS = RT.CFG["processing_delay_ms"]
MASK = RT.CFG["min_elevation_deg"]

# A single ground relay cannot span the route: two legs of at most 3,329 km
# fall short of 7,826 km. A chain of two relays is the minimum, and these are
# real sites on the corridor that satisfy the per-leg range constraint.
RELAY_CHAIN = [
    ("Sydney",       RT.CFG["sydney"]),
    ("Port Moresby", (-9.4438, 147.1803)),
    ("Guam",         (13.4443, 144.7937)),
    ("Tokyo",        RT.CFG["tokyo"]),
]


def latlon_xyz(lat, lon, radius=R):
    la, lo = math.radians(lat), math.radians(lon)
    return (radius * math.cos(la) * math.cos(lo),
            radius * math.cos(la) * math.sin(lo),
            radius * math.sin(la))


def best_single_hop(g_a, g_b, altitude_km):
    """Shortest ground-satellite-ground path, sweeping satellite position.

    Returns the path length in km, or None if no satellite position at this
    altitude can see both endpoints above the elevation mask.
    """
    a = R + altitude_km
    best = None
    for lat in np.linspace(-80, 80, 81):
        for lon in np.linspace(-180, 180, 181):
            s = latlon_xyz(lat, lon, a)
            if RT.elevation_deg(g_a, s) < MASK:
                continue
            if RT.elevation_deg(g_b, s) < MASK:
                continue
            d = RT.dist3(g_a, s) + RT.dist3(s, g_b)
            if best is None or d < best:
                best = d
    return best


def delay_ms(length_km, relays):
    return 1000.0 * length_km / C + relays * PROC_MS


def main():
    syd = latlon_xyz(*RT.CFG["sydney"])
    tok = latlon_xyz(*RT.CFG["tokyo"])
    alt = RT.CFG["altitude_km"]

    print("Sydney to Tokyo, great-circle separation 7,826 km")
    print(f"LEO altitude {alt:.0f} km, elevation mask {MASK:.0f} deg, "
          f"per-hop processing {PROC_MS:.1f} ms\n")

    # ---- A. bent-pipe LEO, no inter-satellite links --------------------
    print("A  Bent-pipe LEO, no ISLs")
    bent = best_single_hop(syd, tok, alt)
    if bent is None:
        phi = math.acos(R * math.cos(math.radians(MASK)) / (R + alt)) - math.radians(MASK)
        span = 2 * R * phi
        print(f"   Not possible. A satellite at {alt:.0f} km spans at most "
              f"{span:,.0f} km at a {MASK:.0f} deg mask,")
        print(f"   against a 7,826 km separation.\n")
    else:
        print(f"   {delay_ms(bent, 1):.2f} ms over {bent:,.0f} km\n")

    # ---- B. ground relay chain -----------------------------------------
    print("B  Ground relay chain, no ISLs")
    phi = math.acos(R * math.cos(math.radians(MASK)) / (R + alt)) - math.radians(MASK)
    span = 2 * R * phi
    print(f"   A single relay is also impossible: two legs of at most "
          f"{span:,.0f} km reach\n   {2*span:,.0f} km, short of the 7,826 km "
          f"separation. Two relays are the minimum.\n")
    print(f"   {'Leg':<32}{'Ground km':>12}{'Path km':>11}")
    print("   " + "-" * 55)
    total, ok = 0.0, True
    for i in range(len(RELAY_CHAIN) - 1):
        n_a, ll_a = RELAY_CHAIN[i]
        n_b, ll_b = RELAY_CHAIN[i + 1]
        g_a, g_b = latlon_xyz(*ll_a), latlon_xyz(*ll_b)
        arc = R * math.acos(max(-1.0, min(1.0,
              sum(p * q for p, q in zip(g_a, g_b)) / R ** 2)))
        leg = best_single_hop(g_a, g_b, alt)
        if leg is None:
            print(f"   {n_a + ' to ' + n_b:<32}{arc:>12,.0f}{'no link':>11}")
            ok = False
            continue
        total += leg
        print(f"   {n_a + ' to ' + n_b:<32}{arc:>12,.0f}{leg:>11,.0f}")

    if ok:
        # 3 satellite hops and 2 intermediate ground stations = 5 relay nodes
        relays = 2 * (len(RELAY_CHAIN) - 1) - 1
        chain_d = delay_ms(total, relays)
        print(f"\n   Total {total:,.0f} km, {2*(len(RELAY_CHAIN)-1)} hops, "
              f"{chain_d:.2f} ms")
        print("   Requires two additional ground stations on the corridor.\n")
    else:
        chain_d = None
        print("\n   Chain not feasible with these sites.\n")

    # ---- B2. the same chain, but with the real constellation ------------
    print("B2 Ground relay chain, using the actual constellation over 200 phases")
    T2 = RT.orbital_period_s(alt)
    ts = np.linspace(0, T2, 200, endpoint=False)
    chain_vals, chain_ok = [], 0
    for t in ts:
        pos, _ = RT.build_graph_at(float(t))
        sats = {k: v for k, v in pos.items() if k.startswith("S")}
        tot, feasible = 0.0, True
        for i in range(len(RELAY_CHAIN) - 1):
            g_a = latlon_xyz(*RELAY_CHAIN[i][1])
            g_b = latlon_xyz(*RELAY_CHAIN[i + 1][1])
            best = None
            for sp in sats.values():
                if RT.elevation_deg(g_a, sp) < MASK:
                    continue
                if RT.elevation_deg(g_b, sp) < MASK:
                    continue
                d = RT.dist3(g_a, sp) + RT.dist3(sp, g_b)
                if best is None or d < best:
                    best = d
            if best is None:
                feasible = False
                break
            tot += best
        if feasible:
            chain_ok += 1
            chain_vals.append(delay_ms(tot, 2 * (len(RELAY_CHAIN) - 1) - 1))

    if chain_vals:
        c_lo, c_hi = np.percentile(chain_vals, [5, 95])
        c_med = statistics.median(chain_vals)
        print(f"   Available at {100*chain_ok/len(ts):.0f} per cent of phases")
        print(f"   {c_med:.2f} ms median, {c_lo:.2f} to {c_hi:.2f} ms\n")
    else:
        c_med = None
        print("   Never available. Each leg requires one satellite visible from")
        print("   both of its endpoints at once, and with 240 satellites the")
        print("   Sydney to Port Moresby window is unoccupied at every sampled")
        print("   phase, so the chain never completes.\n")

    # ---- C. single MEO relay -------------------------------------------
    print("C  Single MEO relay at 3,000 km")
    meo = best_single_hop(syd, tok, 3000.0)
    meo_d = delay_ms(meo, 1)
    print(f"   {meo_d:.2f} ms over {meo:,.0f} km in 2 hops")
    print("   Best case: assumes a satellite at the optimal position.\n")

    # ---- D. meshed LEO with ISLs ---------------------------------------
    print("D  Meshed LEO with inter-satellite links")
    T = RT.orbital_period_s(alt)
    times = np.linspace(0, T, 200, endpoint=False)
    vals, hops = [], []
    for t in times:
        pos, g = RT.build_graph_at(float(t))
        r = RT.route(pos, g)
        if r:
            vals.append(r["total_ms"]); hops.append(r["hops"])
    lo, hi = np.percentile(vals, [5, 95])
    med = statistics.median(vals)
    print(f"   {med:.2f} ms median, {lo:.2f} to {hi:.2f} ms across 200 "
          f"constellation phases")
    print(f"   {statistics.mean(hops):.1f} hops on average, available at "
          f"{100*len(vals)/len(times):.0f} per cent of phases\n")

    # ---- summary --------------------------------------------------------
    print("=" * 63)
    print(f"{'Architecture':<38}{'Hops':>7}{'Delay (ms)':>18}")
    print("-" * 63)
    print(f"{'A  Bent-pipe LEO, no ISLs':<38}{'n/a':>7}{'not possible':>18}")
    if chain_d:
        print(f"{'B  Ground relay, idealised':<38}"
              f"{2*(len(RELAY_CHAIN)-1):>7}{chain_d:>18.2f}")
    if chain_vals:
        print(f"{'B2 Ground relay, real constellation':<38}"
              f"{2*(len(RELAY_CHAIN)-1):>7}{c_med:>18.2f}"
              f"   ({100*chain_ok/len(ts):.0f}% available)")
    else:
        print(f"{'B2 Ground relay, real constellation':<38}"
              f"{'n/a':>7}{'never available':>18}")
    print(f"{'C  Single MEO relay, best case':<38}{2:>7}{meo_d:>18.2f}")
    print(f"{'D  Meshed LEO with ISLs':<38}"
          f"{statistics.mean(hops):>7.1f}{med:>18.2f}")
    print("=" * 63)
    make_figure(chain_d, meo_d, med, lo, hi, c_med if chain_vals else None)

    print("\nNote on comparability. Architectures A, B and C sweep all satellite\n"
          "positions and take the optimum, so they are upper bounds that assume a\n"
          "satellite exactly where it is wanted. Only B2 and D use the actual\n"
          "constellation across 200 phases and are directly comparable.")
    if chain_vals:
        diff = c_med - med
        faster = "faster" if diff > 0 else "slower"
        print(f"\nAgainst the real constellation, the meshed architecture is "
              f"{abs(diff):.2f} ms {faster}\nthan the ground-relay chain, which is "
              f"available at only {100*chain_ok/len(ts):.0f} per cent of phases and\n"
              f"depends on two additional ground stations on a corridor that is\n"
              f"almost entirely ocean.")


def make_figure(chain_d, meo_d, med, lo, hi, c_med):
    """Bar chart of the in-space architectures. Unavailable options are drawn
    as hatched zero-height markers with a label, so that 'impossible' does not
    read as 'very fast'."""
    labels = ["A\nBent-pipe LEO\nno ISLs",
              "B\nGround relay\nidealised",
              "B2\nGround relay\nreal constellation",
              "C\nSingle MEO relay\nbest case",
              "D\nMeshed LEO\nwith ISLs"]
    values = [0.0, chain_d, 0.0 if c_med is None else c_med, meo_d, med]
    avail = [False, True, c_med is not None, True, True]
    colors = ["#bbbbbb", "#9aa7b5", "#bbbbbb", "#7a4fa3", "#1f4e79"]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bars = ax.bar(labels, values, color=colors, width=0.6, alpha=0.92)
    err_x = len(labels) - 1
    ax.errorbar(err_x, med, yerr=[[med - lo], [hi - med]], fmt="none",
                ecolor="0.2", lw=1.5, capsize=7)

    for i, (b, v, ok) in enumerate(zip(bars, values, avail)):
        if ok:
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f} ms",
                    ha="center", fontsize=11.5, fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width() / 2, 1.5,
                    "not\npossible", ha="center", va="bottom",
                    fontsize=10.5, color="#a00000", fontweight="bold")

    ax.axhspan(0, 0.01, color="none")
    ax.set_ylabel("One-way delay, Sydney to Tokyo (ms)", fontsize=11.5)
    ax.set_ylim(0, max(v for v in values if v) * 1.30)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(alpha=0.3, lw=0.5, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("In-space architectures for the same route\n"
                 "Shaded bars are idealised upper bounds that assume an optimally "
                 "placed satellite.\nOnly B2 and D use the actual constellation "
                 "across 200 orbital phases.", fontsize=10.5, pad=12)

    fig.tight_layout()
    fig.savefig("baselines_chart.png", dpi=300, facecolor="white")
    fig.savefig("baselines_chart.tiff", dpi=300, facecolor="white")
    plt.close(fig)
    print("\nWritten: baselines_chart.png / .tiff")


if __name__ == "__main__":
    main()
