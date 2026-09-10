"""Effect of orbital plane count on latency and redundancy.

Holds the total satellite count and the shell altitude constant, varies only
how those satellites are distributed across orbital planes, and measures:

    - end-to-end modelled delay for a fixed Sydney-Tokyo ground pair
    - redundancy, as the number of node-disjoint paths between the two ground
      stations (equivalently the minimum vertex cut)

Both endpoints, altitude, satellite count, link constraints and delay model are
identical across every configuration.
"""

import math
import heapq
import itertools
from collections import defaultdict

import numpy as np

R_EARTH_KM = 6371.0
C_KM_S = 299_792.458

CFG = {
    "altitude_km": 550.0,
    "total_sats": 40,
    "inclination_deg": 53.0,
    "max_isl_km": 2500.0,
    "min_elevation_deg": 10.0,
    "processing_delay_ms": 1.0,
    "sydney": (-33.8688, 151.2093),
    "tokyo": (35.6762, 139.6503),
}


def latlon_to_xyz(lat_deg, lon_deg, radius_km=R_EARTH_KM):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return (radius_km * math.cos(lat) * math.cos(lon),
            radius_km * math.cos(lat) * math.sin(lon),
            radius_km * math.sin(lat))


def build_shell(num_planes, sats_per_plane, altitude_km, inclination_deg):
    """Walker-style shell. Planes are separated in RAAN and phased so that
    satellites in adjacent planes are not at identical in-plane angles."""
    R = R_EARTH_KM + altitude_km
    inc = math.radians(inclination_deg)
    out = []
    for p in range(num_planes):
        raan = 2 * math.pi * p / num_planes
        phase = math.pi * p / max(1, num_planes)   # inter-plane phasing
        for k in range(sats_per_plane):
            theta = 2 * math.pi * k / sats_per_plane + phase
            x, y = R * math.cos(theta), R * math.sin(theta)
            yi, zi = y * math.cos(inc), y * math.sin(inc)
            out.append((x * math.cos(raan) - yi * math.sin(raan),
                        x * math.sin(raan) + yi * math.cos(raan),
                        zi))
    return out


def dist3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def seg_clears_earth(a, b):
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
    if vn == 0:
        return 90.0
    cz = max(-1.0, min(1.0, sum(u[i] * v[i] for i in range(3)) / vn))
    return 90.0 - math.degrees(math.acos(cz))


def build_network(num_planes):
    total, alt = CFG["total_sats"], CFG["altitude_km"]
    if total % num_planes:
        return None
    sats = build_shell(num_planes, total // num_planes, alt, CFG["inclination_deg"])

    pos = {f"S{i}": p for i, p in enumerate(sats)}
    pos["G0"] = latlon_to_xyz(*CFG["sydney"])
    pos["G1"] = latlon_to_xyz(*CFG["tokyo"])

    graph = defaultdict(list)
    ids = list(pos)
    for a, b in itertools.combinations(ids, 2):
        if a.startswith("G") and b.startswith("G"):
            continue
        d = dist3(pos[a], pos[b])
        if a.startswith("G") or b.startswith("G"):
            # A ground station sits exactly at the Earth's radius, so the
            # segment-clears-Earth test is decided by floating-point noise at
            # its own endpoint. The elevation mask is the correct constraint
            # here and already implies an unobstructed line of sight.
            g, s = (a, b) if a.startswith("G") else (b, a)
            if elevation_deg(pos[g], pos[s]) < CFG["min_elevation_deg"]:
                continue
        else:
            if not seg_clears_earth(pos[a], pos[b]):
                continue
            if d > CFG["max_isl_km"]:
                continue
        graph[a].append((b, d))
        graph[b].append((a, d))
    return pos, graph


def shortest_path(pos, graph, src, dst, proc_ms):
    proc_s = proc_ms / 1000.0
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
    total_ms = 1000.0 * length / C_KM_S + relays * proc_ms
    return {"path": path, "hops": len(path) - 1, "length_km": length,
            "total_ms": total_ms}


def node_disjoint_paths(graph, src, dst):
    """Number of internally node-disjoint src-dst paths = min vertex cut.

    Standard node-splitting reduction, then unit-capacity max flow (BFS
    augmenting paths). Ground stations are not split, since they are the
    endpoints.
    """
    cap = defaultdict(lambda: defaultdict(int))

    def add(u, v, c):
        cap[u][v] += c
        cap[v][u] += 0

    for n in graph:
        if n in (src, dst):
            continue
        add(f"{n}_in", f"{n}_out", 1)

    def out_of(n):
        return n if n in (src, dst) else f"{n}_out"

    def into(n):
        return n if n in (src, dst) else f"{n}_in"

    for u in graph:
        for v, _ in graph[u]:
            add(out_of(u), into(v), 1_000_000)

    flow = 0
    while True:
        parent = {src: None}
        q = [src]
        while q and dst not in parent:
            u = q.pop(0)
            for v in list(cap[u]):
                if v not in parent and cap[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        if dst not in parent:
            return flow
        v, bottleneck = dst, math.inf
        while parent[v] is not None:
            bottleneck = min(bottleneck, cap[parent[v]][v])
            v = parent[v]
        v = dst
        while parent[v] is not None:
            cap[parent[v]][v] -= bottleneck
            cap[v][parent[v]] += bottleneck
            v = parent[v]
        flow += bottleneck


def main():
    print("Total satellites held constant at "
          f"{CFG['total_sats']}, altitude {CFG['altitude_km']:.0f} km, "
          f"inclination {CFG['inclination_deg']:.0f} deg")
    print(f"ISL range {CFG['max_isl_km']:.0f} km, elevation mask "
          f"{CFG['min_elevation_deg']:.0f} deg, "
          f"processing {CFG['processing_delay_ms']:.1f} ms/hop")
    print(f"Route: Sydney to Tokyo\n")

    hdr = (f"{'Planes':>7}{'Sats/plane':>12}{'Hops':>6}{'Length km':>12}"
           f"{'Delay ms':>11}{'Disjoint paths':>16}")
    print(hdr)
    print("-" * len(hdr))

    for np_ in [1, 2, 4, 5, 8, 10, 20]:
        net = build_network(np_)
        if net is None:
            continue
        pos, graph = net
        r = shortest_path(pos, graph, "G0", "G1", CFG["processing_delay_ms"])
        k = node_disjoint_paths(graph, "G0", "G1")
        if r is None:
            print(f"{np_:>7}{CFG['total_sats']//np_:>12}{'no path':>6}"
                  f"{'-':>12}{'-':>11}{k:>16}")
        else:
            print(f"{np_:>7}{CFG['total_sats']//np_:>12}{r['hops']:>6}"
                  f"{r['length_km']:>12.0f}{r['total_ms']:>11.2f}{k:>16}")


if __name__ == "__main__":
    main()
