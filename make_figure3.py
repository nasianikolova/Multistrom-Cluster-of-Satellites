"""Figure 3: three-dimensional representations of the constellation.

A. Three LEO orbital planes with a MEO backbone plane.
B. Fifty satellites distributed by Fibonacci sphere initialisation followed by
   repulsion relaxation.
C. A Walker constellation of four single-plane shells.

Every panel carries axis labels in kilometres, tick labels and a legend
distinguishing the shells. Axis ranges are symmetric. The Earth is drawn
translucent with a wireframe at the orbital radius, since at plot scale the
550 km altitude is only 8.6 per cent of the Earth's radius and would otherwise
appear to rest on the surface.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

R_EARTH_KM = 6371.0
LEO_COLOR = "#1f4e79"
MEO_COLOR = "#e07b00"
RING_COLOR = "0.55"


def shell(num_planes, sats_per_plane, altitude_km, inclination_deg, f=1):
    """Walker-style shell with inter-plane phasing."""
    a = R_EARTH_KM + altitude_km
    inc = math.radians(inclination_deg)
    pts, rings = [], []
    for p in range(num_planes):
        raan = 2 * math.pi * p / num_planes
        ring = []
        for k in range(200):
            th = 2 * math.pi * k / 200
            x, y = a * math.cos(th), a * math.sin(th)
            yi, zi = y * math.cos(inc), y * math.sin(inc)
            ring.append((x * math.cos(raan) - yi * math.sin(raan),
                         x * math.sin(raan) + yi * math.cos(raan), zi))
        rings.append(np.array(ring))
        for k in range(sats_per_plane):
            th = 2 * math.pi * k / sats_per_plane + 2 * math.pi * f * p / (num_planes * sats_per_plane)
            x, y = a * math.cos(th), a * math.sin(th)
            yi, zi = y * math.cos(inc), y * math.sin(inc)
            pts.append((x * math.cos(raan) - yi * math.sin(raan),
                        x * math.sin(raan) + yi * math.cos(raan), zi))
    return np.array(pts), rings


def fibonacci_repulsion(n=50, altitude_km=550.0, steps=1000, step_size=0.01, seed=2):
    """Fibonacci-sphere initialisation followed by inverse-square repulsion."""
    rng = np.random.default_rng(seed)
    a = R_EARTH_KM + altitude_km
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    pts = np.column_stack([np.sin(phi) * np.cos(theta),
                           np.sin(phi) * np.sin(theta),
                           np.cos(phi)]) * a
    for s in range(steps):
        eff = step_size * (1 - s / (steps * 50))
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dist, np.inf)
        force = (d / dist[:, :, None] ** 3).sum(axis=1)
        pts = pts + eff * force * a
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * a
    return pts


def draw_earth(ax, alpha=0.28):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(R_EARTH_KM * np.cos(u) * np.sin(v),
                    R_EARTH_KM * np.sin(u) * np.sin(v),
                    R_EARTH_KM * np.cos(v),
                    color="#9dc3e6", alpha=alpha, linewidth=0, shade=True,
                    zorder=0)


def style(ax, limit, title, note):
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (km)", fontsize=9, labelpad=8)
    ax.set_ylabel("y (km)", fontsize=9, labelpad=8)
    ax.set_zlabel("z (km)", fontsize=9, labelpad=10)
    # Ticks placed inboard of the axis ends so that the x and y labels do not
    # collide at the near corner of the 3D box.
    ticks = np.array([-limit / 2.0, 0.0, limit / 2.0])
    ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
    ax.tick_params(labelsize=7.5, pad=3)
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.text2D(0.0, -0.06, note, transform=ax.transAxes, fontsize=8.2,
              color="0.30", va="top")
    ax.view_init(elev=22, azim=38)


def main():
    fig = plt.figure(figsize=(15.0, 5.4))

    # ---------------- A: LEO shells plus MEO backbone ----------------
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw_earth(ax)
    leo, leo_rings = shell(3, 10, 550.0, 53.0)
    meo, meo_rings = shell(1, 5, 3000.0, 45.0)
    for r in leo_rings:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color=RING_COLOR, lw=0.7, alpha=0.8)
    for r in meo_rings:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color=MEO_COLOR, lw=0.9, alpha=0.7)
    ax.scatter(leo[:, 0], leo[:, 1], leo[:, 2], s=20, color=LEO_COLOR,
               depthshade=False, label="LEO satellites, 550 km")
    ax.scatter(meo[:, 0], meo[:, 1], meo[:, 2], s=60, marker="^", color=MEO_COLOR,
               depthshade=False, label="MEO backbone, 3,000 km")
    style(ax, 9600, "A   LEO shells with MEO backbone",
          "3 LEO planes \u00d7 10 satellites at 550 km, 53\u00b0 inclination;\n"
          "1 MEO plane \u00d7 5 satellites at 3,000 km, 45\u00b0 inclination")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.9,
              bbox_to_anchor=(1.10, 0.98))

    # ---------------- B: Fibonacci plus repulsion ----------------
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    draw_earth(ax)
    pts = fibonacci_repulsion()
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=26, color=LEO_COLOR,
               depthshade=False, label="50 satellites, 550 km")
    # An explicit wireframe at the orbital radius makes the 550 km shell
    # visible, so the satellites read as orbiting rather than resting.
    a_orb = R_EARTH_KM + 550.0
    uu, vv = np.mgrid[0:2 * np.pi:19j, 0:np.pi:10j]
    ax.plot_wireframe(a_orb * np.cos(uu) * np.sin(vv),
                      a_orb * np.sin(uu) * np.sin(vv),
                      a_orb * np.cos(vv),
                      color=LEO_COLOR, lw=0.35, alpha=0.30)
    style(ax, 7600, "B   Fibonacci distribution with repulsion",
          "50 satellites at 550 km altitude. The wireframe marks the\n"
          "orbital shell at radius 6,921 km, above the 6,371 km surface")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.9,
              bbox_to_anchor=(1.08, 0.98))

    # ---------------- C: Walker constellation ----------------
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw_earth(ax)
    shells = [(1, 12, 550.0, 60.0, LEO_COLOR, "o", 20, "LEO 550 km, 60\u00b0: 12/1/0"),
              (1, 21, 3000.0, 60.0, MEO_COLOR, "^", 46, "MEO 3,000 km, 60\u00b0: 21/1/0"),
              (1, 12, 550.0, 0.0, "#2e9e5b", "s", 24, "LEO 550 km, 0\u00b0: 12/1/0"),
              (1, 12, 550.0, 90.0, "#8a4fbd", "D", 22, "LEO 550 km, 90\u00b0: 12/1/0")]
    for np_, spp, alt, inc, col, mk, sz, lab in shells:
        pts, rings = shell(np_, spp, alt, inc)
        for r in rings:
            ax.plot(r[:, 0], r[:, 1], r[:, 2], color=col, lw=0.7, alpha=0.45)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=sz, marker=mk, color=col,
                   depthshade=False, label=lab)
    style(ax, 9600, "C   Walker constellation, four shells",
          "Walker notation i: T/P/F. Total 57 satellites in four\n"
          "single-plane shells; F = 0 since each shell has one plane")
    ax.legend(loc="upper right", fontsize=7.0, framealpha=0.9,
              bbox_to_anchor=(1.13, 1.00))

    fig.subplots_adjust(left=0.045, right=0.975, top=0.93, bottom=0.12, wspace=0.16)
    fig.savefig("Figure3_3d_representations.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig("Figure3_3d_representations.tiff", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Panel A: 30 LEO satellites at 550 km, 5 MEO satellites at 3,000 km")
    print("Panel B: 50 satellites at 550 km, Fibonacci init, 1000 repulsion steps")
    print("Panel C: 12 + 21 + 12 + 12 = 57 satellites across four single-plane shells")
    print("\nWritten: Figure3_3d_representations.png / .tiff")


if __name__ == "__main__":
    main()
