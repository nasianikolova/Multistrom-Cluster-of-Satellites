"""Figure 5: routing state over time on a fixed constellation.

No propulsion, no delta-v, no orbit changes. Satellites move along their orbits
under Keplerian circular motion and the Earth rotates beneath them. At each
epoch the connectivity graph is rebuilt from scratch and the route is
recomputed. What changes over time is the route, not the constellation.

Reported per epoch: whether a path exists, its hop count, physical length and
modelled delay, and whether the route differs from the previous epoch, which is
a handover.
"""

import math
import heapq
import itertools
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH_KM = 6371.0
C_KM_S = 299_792.458
MU = 398_600.4418          # km^3/s^2, Earth gravitational parameter
EARTH_ROT_S = 86_164.0905  # sidereal day

CFG = {
    "altitude_km": 550.0,
    "num_planes": 12,
    "sats_per_plane": 20,
    "inclination_deg": 53.0,
    "walker_f": 1,
    "max_isl_km": 2500.0,
    "min_elevation_deg": 10.0,
    "processing_delay_ms": 1.0,
    "sydney": (-33.8688, 151.2093),
    "tokyo": (35.6762, 139.6503),
}


def orbital_period_s(altitude_km):
    a = R_EARTH_KM + altitude_km
    return 2 * math.pi * math.sqrt(a ** 3 / MU)


def sat_positions(t_s):
    """Satellite positions at time t. Circular Keplerian motion, no propulsion."""
    a = R_EARTH_KM + CFG["altitude_km"]
    inc = math.radians(CFG["inclination_deg"])
    n = 2 * math.pi / orbital_period_s(CFG["altitude_km"])   # mean motion
    P, S, f = CFG["num_planes"], CFG["sats_per_plane"], CFG["walker_f"]

    out = []
    for p in range(P):
        raan = 2 * math.pi * p / P
        for k in range(S):
            theta = 2 * math.pi * k / S + 2 * math.pi * f * p / (P * S) + n * t_s
            x, y = a * math.cos(theta), a * math.sin(theta)
            yi, zi = y * math.cos(inc), y * math.sin(inc)
            out.append((x * math.cos(raan) - yi * math.sin(raan),
                        x * math.sin(raan) + yi * math.cos(raan),
                        zi))
    return out


def ground_position(lat_deg, lon_deg, t_s):
    """Ground station in the inertial frame; the Earth rotates beneath the orbits."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg) + 2 * math.pi * t_s / EARTH_ROT_S
    return (R_EARTH_KM * math.cos(lat) * math.cos(lon),
            R_EARTH_KM * math.cos(lat) * math.sin(lon),
            R_EARTH_KM * math.sin(lat))


def dist3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def clears_earth(a, b):
    ax, ay, az = a
    dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
    dd = dx * dx + dy * dy + dz * dz
    if dd == 0:
        return True
    t = max(0.0, min(1.0, -(ax * dx + ay * dy + az * dz) / dd))
    px, py, pz = ax + t * dx, ay + t * dy, az + t * dz
    return math.sqrt(px * px + py * py + pz * pz) >= R_EARTH_KM


def elevation_deg(ground, sat):
    gn = math.sqrt(sum(c * c for c in ground))
    u = [c / gn for c in ground]
    v = [sat[i] - ground[i] for i in range(3)]
    vn = math.sqrt(sum(c * c for c in v))
    cz = max(-1.0, min(1.0, sum(u[i] * v[i] for i in range(3)) / vn))
    return 90.0 - math.degrees(math.acos(cz))


def build_graph_at(t_s):
    """Rebuilt from scratch at every epoch. No state carries over."""
    pos = {f"S{i}": p for i, p in enumerate(sat_positions(t_s))}
    pos["G0"] = ground_position(*CFG["sydney"], t_s)
    pos["G1"] = ground_position(*CFG["tokyo"], t_s)

    graph = defaultdict(list)
    for a, b in itertools.combinations(pos, 2):
        if a[0] == "G" and b[0] == "G":
            continue
        d = dist3(pos[a], pos[b])
        if a[0] == "G" or b[0] == "G":
            g, s = (a, b) if a[0] == "G" else (b, a)
            if elevation_deg(pos[g], pos[s]) < CFG["min_elevation_deg"]:
                continue
        else:
            if d > CFG["max_isl_km"] or not clears_earth(pos[a], pos[b]):
                continue
        graph[a].append((b, d))
        graph[b].append((a, d))
    return pos, graph


def route(pos, graph, src="G0", dst="G1"):
    proc_s = CFG["processing_delay_ms"] / 1000.0
    dist = {n: math.inf for n in pos}
    prev = {n: None for n in pos}
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, dd in graph[u]:
            nd = d + dd / C_KM_S + proc_s
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] == math.inf:
        return None
    path, n = [], dst
    while n is not None:
        path.append(n)
        n = prev[n]
    path.reverse()
    length = sum(dist3(pos[path[i]], pos[path[i + 1]]) for i in range(len(path) - 1))
    relays = len(path) - 2
    return {"path": path, "hops": len(path) - 1, "length_km": length,
            "total_ms": 1000.0 * length / C_KM_S + relays * CFG["processing_delay_ms"]}


def main():
    T = orbital_period_s(CFG["altitude_km"])
    n_epochs = 120
    times = np.linspace(0, T, n_epochs, endpoint=False)
    total = CFG["num_planes"] * CFG["sats_per_plane"]

    print(f"Constellation: {total} satellites, {CFG['num_planes']} planes x "
          f"{CFG['sats_per_plane']}, {CFG['altitude_km']:.0f} km, "
          f"{CFG['inclination_deg']:.0f} deg inclination, Walker f={CFG['walker_f']}")
    print(f"Orbital period: {T/60:.1f} min. Sampling {n_epochs} epochs over one period "
          f"({T/n_epochs:.0f} s apart).")
    print("Graph rebuilt from scratch at every epoch.\n")

    delays, hops, avail, paths = [], [], [], []
    for t in times:
        pos, g = build_graph_at(float(t))
        r = route(pos, g)
        if r is None:
            avail.append(0); delays.append(np.nan); hops.append(np.nan); paths.append(None)
        else:
            avail.append(1); delays.append(r["total_ms"]); hops.append(r["hops"])
            paths.append(tuple(r["path"]))

    delays, hops = np.array(delays), np.array(hops)
    ok = ~np.isnan(delays)
    changes = sum(1 for i in range(1, len(paths))
                  if paths[i] is not None and paths[i - 1] is not None
                  and paths[i] != paths[i - 1])

    print(f"Path availability      : {100*np.mean(avail):.1f}% of epochs")
    print(f"End-to-end delay       : mean {np.nanmean(delays):.2f} ms, "
          f"min {np.nanmin(delays):.2f}, max {np.nanmax(delays):.2f}, "
          f"sd {np.nanstd(delays):.2f}")
    print(f"Hop count              : min {int(np.nanmin(hops))}, "
          f"max {int(np.nanmax(hops))}, mean {np.nanmean(hops):.2f}")
    print(f"Route changes          : {changes} over {ok.sum()} connected epochs "
          f"({100*changes/max(1,ok.sum()-1):.0f}% of consecutive pairs)")
    print(f"Distinct routes used   : {len(set(p for p in paths if p))}")
    print(f"Mean route lifetime    : "
          f"{(T/60)/max(1,changes):.2f} min before the path changes")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    tm = times / 60.0
    ax1.plot(tm, delays, color="#1f4e79", lw=1.6)
    ax1.scatter(tm[~ok], np.full((~ok).sum(), np.nanmin(delays)), s=14,
                color="#c00000", zorder=3, label="no path available")
    ax1.set_ylabel("End-to-end delay (ms)")
    ax1.grid(alpha=0.3, lw=0.4)
    ax1.set_title(f"Routing state over one orbital period ({T/60:.0f} min)\n"
                  f"{total} satellites, {CFG['num_planes']} planes, "
                  f"{CFG['altitude_km']:.0f} km, Sydney\u2013Tokyo", fontsize=10)
    if (~ok).any():
        ax1.legend(fontsize=8)

    ax2.step(tm, hops, where="post", color="#e07b00", lw=1.5)
    ax2.set_ylabel("Hops")
    ax2.set_xlabel("Time since epoch (minutes)")
    ax2.set_yticks(range(int(np.nanmin(hops)), int(np.nanmax(hops)) + 1))
    ax2.grid(alpha=0.3, lw=0.4)

    fig.savefig("Figure5_routing_over_time.png", dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nWritten: Figure5_routing_over_time.png")


if __name__ == "__main__":
    main()
