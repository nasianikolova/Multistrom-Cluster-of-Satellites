"""Render the publication version of Figure 6 as a single three-panel image.

Panel A  LEO-only topology and the selected route.
Panel B  LEO + MEO topology, showing BOTH the route the router selects and the
         backbone route that is available but rejected, so the geometric
         trade-off (fewer hops, longer path) is visible directly.
Panel C  Total end-to-end delay against per-hop processing delay, with the
         break-even point marked.

One file, one figure number, one caption.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constellation as F

R = F.R_EARTH_KM


def draw_topology(ax, sim, selected, alternative=None, title=""):
    ax.add_patch(plt.Circle((0, 0), R, color="#c8d8e8", zorder=0))

    for a in sim.nodes:
        for b_id, _ in sim.graph[a.id]:
            b = next(n for n in sim.nodes if n.id == b_id)
            ax.plot([a.pos[0], b.pos[0]], [a.pos[1], b.pos[1]],
                    color="0.90", lw=0.4, zorder=1)

    for t, kw in [("leo", dict(s=12, c="#1f4e79", marker="o", label="LEO satellites")),
                  ("meo", dict(s=55, c="#e07b00", marker="^", label="MEO backbone")),
                  ("ground", dict(s=70, c="#c00000", marker="s", label="Ground stations"))]:
        pts = [n.pos for n in sim.nodes if n.type == t]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], zorder=4, **kw)

    def draw_route(path, color, ls, lw, label, z):
        pos = {n.id: n.pos for n in sim.nodes}
        for i in range(len(path) - 1):
            ax.plot([pos[path[i]][0], pos[path[i + 1]][0]],
                    [pos[path[i]][1], pos[path[i + 1]][1]],
                    color=color, ls=ls, lw=lw, zorder=z,
                    label=label if i == 0 else None)

    if alternative:
        draw_route(alternative, "#e07b00", "--", 2.0, "Backbone route (rejected)", 5)
    draw_route(selected, "#c00000", "-", 2.4, "Selected route", 6)

    ax.set_title(title, fontsize=10, pad=8)
    lim = 1.12 * (R + max(s.altitude_km for s in sim.shells))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("km", fontsize=8); ax.set_ylabel("km", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)


def main():
    pm = F.CONFIG["processing_delay_ms"]
    sims, rows = F.run_table(pm)

    sim_a = sims["A  LEO only (40 sats)"]
    _, path_a = sim_a.shortest_path("G0", "G1")
    m_a = sim_a.path_metrics(path_a)

    sim_b = sims["B  LEO 40 + MEO backbone 5"]
    _, path_b = sim_b.shortest_path("G0", "G1")
    m_b = sim_b.path_metrics(path_b)

    # The backbone route that exists but is not selected at this processing delay
    _, up = sim_b.shortest_path("G0", "M0")
    _, down = sim_b.shortest_path("M0", "G1")
    path_bb = up + down[1:]
    m_bb = sim_b.path_metrics(path_bb)

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.12], wspace=0.28)

    draw_topology(fig.add_subplot(gs[0]), sim_a, path_a,
                  title=f"A   LEO only, 40 satellites\n"
                        f"{m_a['hops']} hops, {m_a['length_km']:.0f} km, "
                        f"{m_a['total_ms']:.2f} ms")

    draw_topology(fig.add_subplot(gs[1]), sim_b, path_b, alternative=path_bb,
                  title=f"B   LEO 40 + MEO backbone, 5 satellites\n"
                        f"selected {m_b['hops']} hops / {m_b['length_km']:.0f} km  ·  "
                        f"backbone {m_bb['hops']} hops / {m_bb['length_km']:.0f} km")

    ax = fig.add_subplot(gs[2])
    sweep = F.crossover_sweep(np.linspace(0.0, 15.0, 301))
    xo = F.find_crossover(sweep)
    ax.plot(sweep["proc_ms"], sweep["A"], color="#1f4e79", lw=2.2,
            label="LEO only (40)")
    ax.plot(sweep["proc_ms"], sweep["B"], color="#e07b00", lw=2.2,
            label="LEO 40 + MEO backbone")
    ax.plot(sweep["proc_ms"], sweep["C"], color="0.55", lw=1.3, ls="--",
            label="LEO only (45), control")
    if xo is not None:
        ax.axvline(xo, color="0.25", ls=":", lw=1.1)
        ax.annotate(f"break-even\n{xo:.2f} ms/hop",
                    xy=(xo, 57), xytext=(xo + 1.6, 44), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="0.25"))
    ax.set_xlabel("Per-hop processing delay (ms)", fontsize=8)
    ax.set_ylabel("End-to-end delay, Sydney–Tokyo (ms)", fontsize=8)
    ax.set_title("C   Delay against per-hop processing cost", fontsize=10, pad=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=7, loc="upper left")

    fig.savefig("Figure4_controlled_comparison.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig("Figure4_controlled_comparison.tiff", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Panel A :", " -> ".join(path_a))
    print(f"          {m_a['hops']} hops, {m_a['length_km']:.1f} km, {m_a['total_ms']:.2f} ms")
    print("Panel B selected :", " -> ".join(path_b))
    print(f"          {m_b['hops']} hops, {m_b['length_km']:.1f} km, {m_b['total_ms']:.2f} ms")
    print("Panel B backbone :", " -> ".join(path_bb))
    print(f"          {m_bb['hops']} hops, {m_bb['length_km']:.1f} km, {m_bb['total_ms']:.2f} ms")
    print(f"Panel C break-even : {xo:.2f} ms/hop")
    print("Written: Figure4_controlled_comparison.png / .tiff")


if __name__ == "__main__":
    main()
