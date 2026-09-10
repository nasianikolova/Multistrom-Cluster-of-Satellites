"""
FAROS - constellation simulation core.

Two-dimensional constellation model with shortest-path routing.

The connectivity graph is rebuilt after any change to node positions, and
shortest_path() raises rather than routing on a stale graph.

A link is admitted only if it satisfies all of: Earth occultation, a minimum
ground elevation angle, a range limit for inter-satellite links, and a link
budget threshold.

End-to-end delay is decomposed into propagation, transmission, processing,
switching and queueing terms. Per-node terms are charged only at intermediate
relay nodes; the destination does not forward the packet and is not charged.

Produces Table 2 of the manuscript.

Usage:  python constellation.py
"""
import math
import heapq
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Constants and configuration (everything the reviewer needs to reproduce)
# --------------------------------------------------------------------------
R_EARTH_KM = 6371.0
C_KM_S = 299_792.458

CONFIG = {
    "leo_altitude_km": 550.0,
    "meo_altitude_km": 3000.0,
    "n_leo": 40,
    "n_meo": 5,
    "max_isl_leo_leo_km": 2500.0,   # LEO-LEO inter-satellite link budget limit
    "max_link_meo_km": 9000.0,      # links involving a MEO node
    "min_elevation_deg": 10.0,      # ground station mask angle
    "processing_delay_ms": 1.0,     # baseline per-hop on-board processing
    "packet_bits": 12000.0,         # 1500-byte packet (Ethernet MTU)
    "link_rate_bps": 1.0e9,         # 1 Gbps inter-satellite link
    "switching_delay_ms": 0.002,    # switching fabric, per relay node
    "queueing_delay_ms": 0.0,       # unloaded-network assumption (see Methods)
    "src_lat_lon": (-33.8688, 151.2093),   # Sydney
    "dst_lat_lon": (35.6762, 139.6503),    # Tokyo
}


def central_angle_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    return math.degrees(math.acos(
        math.sin(p1) * math.sin(p2) + math.cos(p1) * math.cos(p2) * math.cos(dl)
    ))


SEPARATION_DEG = central_angle_deg(*CONFIG["src_lat_lon"], *CONFIG["dst_lat_lon"])
GREAT_CIRCLE_KM = math.radians(SEPARATION_DEG) * R_EARTH_KM


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def polar_to_cartesian(radius_km: float, angle_rad: float) -> Tuple[float, float]:
    return (radius_km * math.cos(angle_rad), radius_km * math.sin(angle_rad))


def euclidean_distance(p1, p2) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def segment_min_dist_to_origin(p1, p2) -> float:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(x1, y1)
    t = -(x1 * dx + y1 * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(x1 + t * dx, y1 + t * dy)


def line_of_sight(p1, p2, earth_radius_km: float = R_EARTH_KM) -> bool:
    """True if the straight segment does not cut the Earth's disc."""
    return segment_min_dist_to_origin(p1, p2) >= earth_radius_km


def elevation_angle_deg(ground_pos, sat_pos) -> float:
    """Elevation of sat above the local horizon at the ground station."""
    gx, gy = ground_pos
    gnorm = math.hypot(gx, gy)
    ux, uy = gx / gnorm, gy / gnorm            # local zenith unit vector
    vx, vy = sat_pos[0] - gx, sat_pos[1] - gy  # ground -> satellite
    vnorm = math.hypot(vx, vy)
    if vnorm == 0:
        return 90.0
    cos_zenith = (ux * vx + uy * vy) / vnorm
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    return 90.0 - math.degrees(math.acos(cos_zenith))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class Node:
    id: str
    pos: Tuple[float, float]
    type: str          # 'leo' | 'meo' | 'ground'


@dataclass
class Shell:
    label: str         # 'leo' or 'meo'
    altitude_km: float
    num_sats: int
    phase_offset_deg: float = 0.0


class Simulation:
    """2D planar constellation model.

    The graph is invalidated on every topology change. shortest_path() refuses
    to run on a stale graph - this is what prevents the original notebook bug
    from ever recurring.
    """

    def __init__(self, shells: List[Shell], ground_angles_deg: List[float],
                 processing_delay_s: float = 0.001,
                 max_isl_leo_leo_km: float = 2500.0,
                 max_link_meo_km: float = 9000.0,
                 min_elevation_deg: float = 10.0):
        self.shells = list(shells)
        self.ground_angles_deg = list(ground_angles_deg)
        self.processing_delay_s = processing_delay_s
        self.max_isl_leo_leo_km = max_isl_leo_leo_km
        self.max_link_meo_km = max_link_meo_km
        self.min_elevation_deg = min_elevation_deg

        self.nodes: List[Node] = []
        self.graph: Dict[str, List[Tuple[str, float]]] = {}
        self._graph_dirty = True

        self.build_nodes()
        self.build_graph()

    # ---- topology -------------------------------------------------------
    def build_nodes(self):
        nodes = []
        for shell in self.shells:
            radius = R_EARTH_KM + shell.altitude_km
            prefix = "L" if shell.label == "leo" else "M"
            for i in range(shell.num_sats):
                ang = 2 * math.pi * i / shell.num_sats \
                      + math.radians(shell.phase_offset_deg)
                nodes.append(Node(id=f"{prefix}{i}",
                                  pos=polar_to_cartesian(radius, ang),
                                  type=shell.label))
        for idx, adeg in enumerate(self.ground_angles_deg):
            nodes.append(Node(id=f"G{idx}",
                              pos=polar_to_cartesian(R_EARTH_KM, math.radians(adeg)),
                              type="ground"))
        self.nodes = nodes
        self._graph_dirty = True

    def move_node(self, node_id: str, new_pos: Tuple[float, float]):
        """Any position change marks the graph stale. This is the fix."""
        for n in self.nodes:
            if n.id == node_id:
                n.pos = new_pos
                self._graph_dirty = True
                return
        raise KeyError(node_id)

    def _link_allowed(self, a: Node, b: Node) -> Optional[float]:
        """Return link distance if a-b is a valid link, else None."""
        if a.type == "ground" and b.type == "ground":
            return None
        if not line_of_sight(a.pos, b.pos):
            return None
        d = euclidean_distance(a.pos, b.pos)

        if "ground" in (a.type, b.type):
            g, s = (a, b) if a.type == "ground" else (b, a)
            if elevation_angle_deg(g.pos, s.pos) < self.min_elevation_deg:
                return None
            return d

        if "meo" in (a.type, b.type):
            return d if d <= self.max_link_meo_km else None

        return d if d <= self.max_isl_leo_leo_km else None

    def build_graph(self):
        graph = {n.id: [] for n in self.nodes}
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1:]:
                d = self._link_allowed(a, b)
                if d is None:
                    continue
                # Routing weight = propagation + one processing step.
                # Adding proc to every edge shifts every path by exactly one
                # constant term, so route selection is unaffected; the reported
                # total subtracts it back out (see path_metrics).
                w = d / C_KM_S + self.processing_delay_s
                graph[a.id].append((b.id, w))
                graph[b.id].append((a.id, w))
        self.graph = graph
        self._graph_dirty = False

    # ---- routing --------------------------------------------------------
    def shortest_path(self, source: str, target: str):
        if self._graph_dirty:
            raise RuntimeError(
                "Graph is stale: the topology changed after the last "
                "build_graph(). Call build_graph() before routing."
            )
        dist = {n.id: math.inf for n in self.nodes}
        prev = {n.id: None for n in self.nodes}
        dist[source] = 0.0
        pq = [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == target:
                break
            if d > dist[u]:
                continue
            for v, w in self.graph[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[target] == math.inf:
            return None, []
        path, node = [], target
        while node is not None:
            path.append(node)
            node = prev[node]
        return dist[target], path[::-1]

    # ---- metrics --------------------------------------------------------
    def path_metrics(self, path: List[str], processing_delay_s=None) -> dict:
        """Full end-to-end delay decomposition for a path.

        T_e2e = SUM over links ( d_i / c  +  L / R )
              + SUM over relay nodes ( t_proc + t_switch + t_queue )

        Links carry propagation and transmission (serialisation). Only
        intermediate nodes relay, so the destination is charged no per-node
        term - the original notebook charged every edge, over-counting by one
        processing step on every path.
        """
        if processing_delay_s is None:
            processing_delay_s = self.processing_delay_s
        if not path:
            return {k: None for k in ("path", "hops", "relays", "length_km",
                                      "prop_ms", "trans_ms", "proc_ms",
                                      "switch_ms", "queue_ms", "total_ms")}
        pos = {n.id: n.pos for n in self.nodes}
        length_km = sum(euclidean_distance(pos[path[i]], pos[path[i + 1]])
                        for i in range(len(path) - 1))
        hops = len(path) - 1                      # links traversed
        relays = max(0, len(path) - 2)            # nodes that actually forward

        prop_ms = 1000.0 * length_km / C_KM_S
        trans_ms = 1000.0 * hops * CONFIG["packet_bits"] / CONFIG["link_rate_bps"]
        proc_ms = 1000.0 * relays * processing_delay_s
        switch_ms = relays * CONFIG["switching_delay_ms"]
        queue_ms = relays * CONFIG["queueing_delay_ms"]

        return {"path": path, "hops": hops, "relays": relays,
                "length_km": length_km, "prop_ms": prop_ms,
                "trans_ms": trans_ms, "proc_ms": proc_ms,
                "switch_ms": switch_ms, "queue_ms": queue_ms,
                "total_ms": prop_ms + trans_ms + proc_ms + switch_ms + queue_ms}


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
def make_scenarios(processing_delay_s: float):
    cfg = CONFIG
    ground = [0.0, SEPARATION_DEG]
    common = dict(ground_angles_deg=ground,
                  processing_delay_s=processing_delay_s,
                  max_isl_leo_leo_km=cfg["max_isl_leo_leo_km"],
                  max_link_meo_km=cfg["max_link_meo_km"],
                  min_elevation_deg=cfg["min_elevation_deg"])

    leo = Shell("leo", cfg["leo_altitude_km"], cfg["n_leo"])
    meo = Shell("meo", cfg["meo_altitude_km"], cfg["n_meo"])
    leo_plus = Shell("leo", cfg["leo_altitude_km"], cfg["n_leo"] + cfg["n_meo"])

    return {
        "A  LEO only (40 sats)":            Simulation([leo], **common),
        "B  LEO 40 + MEO backbone 5":       Simulation([leo, meo], **common),
        "C  LEO only, 45 sats (control)":   Simulation([leo_plus], **common),
    }


def run_table(processing_delay_ms: float):
    sims = make_scenarios(processing_delay_ms / 1000.0)
    rows = []
    for name, sim in sims.items():
        total, path = sim.shortest_path("G0", "G1")
        m = sim.path_metrics(path)
        m["scenario"] = name
        m["n_sats"] = sum(s.num_sats for s in sim.shells)
        rows.append(m)
    return sims, rows


def print_table(rows, processing_delay_ms):
    hdr = (f"{'Scenario':<32}{'Sats':>5}{'Hops':>5}{'Length km':>11}"
           f"{'Prop':>9}{'Trans':>8}{'Proc':>8}{'Switch':>8}{'Queue':>8}{'TOTAL':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['scenario']:<32}{r['n_sats']:>5}{r['hops']:>5}"
              f"{r['length_km']:>11.1f}{r['prop_ms']:>9.2f}{r['trans_ms']:>8.3f}"
              f"{r['proc_ms']:>8.2f}{r['switch_ms']:>8.3f}{r['queue_ms']:>8.2f}"
              f"{r['total_ms']:>9.2f}")
    print("-" * len(hdr))
    for r in rows:
        print(f"  {r['scenario']}  path: {' -> '.join(r['path'])}")


def crossover_sweep(proc_ms_values):
    """Total delay of A and B as a function of per-hop processing delay."""
    out = {"proc_ms": [], "A": [], "B": [], "C": []}
    for pm in proc_ms_values:
        sims = make_scenarios(pm / 1000.0)
        out["proc_ms"].append(pm)
        for key, name in [("A", "A  LEO only (40 sats)"),
                          ("B", "B  LEO 40 + MEO backbone 5"),
                          ("C", "C  LEO only, 45 sats (control)")]:
            sim = sims[name]
            _, path = sim.shortest_path("G0", "G1")
            t = sim.path_metrics(path)["total_ms"]
            # A disconnected topology is not "zero delay" - it is unusable.
            out[key].append(math.inf if t is None else t)
    return out


def find_crossover(sweep):
    """Smallest processing delay at which B beats A, by linear interpolation."""
    p, a, b = sweep["proc_ms"], sweep["A"], sweep["B"]
    # Below the crossover both scenarios select the SAME LEO path, so the
    # difference is exactly zero, not negative. Look for the first point where
    # B is strictly faster, then interpolate back over the last zero-diff step.
    for i in range(1, len(p)):
        if b[i] < a[i] - 1e-9:
            d0, d1 = a[i - 1] - b[i - 1], a[i] - b[i]
            if d1 == d0:
                return p[i]
            return p[i - 1] + (p[i] - p[i - 1]) * (-d0) / (d1 - d0)
    return None


def isl_sensitivity(isl_values, proc_grid):
    """How the crossover point moves with the LEO inter-satellite link range.

    Restores CONFIG afterwards so the function has no side effects.
    """
    saved = CONFIG["max_isl_leo_leo_km"]
    rows = []
    try:
        for isl in isl_values:
            CONFIG["max_isl_leo_leo_km"] = isl
            _, r = run_table(CONFIG["processing_delay_ms"])
            sweep = crossover_sweep(proc_grid)
            rows.append({"isl_km": isl,
                         "hops_A": r[0]["hops"],
                         "length_A_km": r[0]["length_km"],
                         "hops_B": r[1]["hops"],
                         "crossover_ms": find_crossover(sweep)})
    finally:
        CONFIG["max_isl_leo_leo_km"] = saved
    return rows


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_topology(sim, path, title, outfile):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.add_patch(plt.Circle((0, 0), R_EARTH_KM, color="lightsteelblue", alpha=0.7))

    for a in sim.nodes:
        for b_id, _ in sim.graph[a.id]:
            b = next(n for n in sim.nodes if n.id == b_id)
            ax.plot([a.pos[0], b.pos[0]], [a.pos[1], b.pos[1]],
                    color="0.88", linewidth=0.5, zorder=1)

    styles = {"leo": dict(s=18, c="tab:blue", label="LEO satellites"),
              "meo": dict(s=70, c="tab:orange", marker="^", label="MEO backbone"),
              "ground": dict(s=90, c="tab:red", marker="s", label="Ground stations")}
    for t, st in styles.items():
        pts = [n.pos for n in sim.nodes if n.type == t]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], zorder=3, **st)

    for i in range(len(path) - 1):
        a = next(n for n in sim.nodes if n.id == path[i])
        b = next(n for n in sim.nodes if n.id == path[i + 1])
        ax.plot([a.pos[0], b.pos[0]], [a.pos[1], b.pos[1]],
                color="tab:red", linewidth=2.5, zorder=4)

    m = sim.path_metrics(path)
    ax.set_title(f"{title}\n{m['hops']} hops | {m['length_km']:.0f} km | "
                 f"{m['total_ms']:.2f} ms total", fontsize=11)
    limit = 1.15 * (R_EARTH_KM + max(s.altitude_km for s in sim.shells))
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    ax.set_xlabel("km"); ax.set_ylabel("km"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(outfile, dpi=160); plt.close(fig)


def plot_crossover(sweep, crossover, outfile):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sweep["proc_ms"], sweep["A"], label="A: LEO only (40)", lw=2)
    ax.plot(sweep["proc_ms"], sweep["B"], label="B: LEO 40 + MEO backbone", lw=2)
    ax.plot(sweep["proc_ms"], sweep["C"], label="C: LEO only (45), control",
            lw=1.5, ls="--", color="gray")
    if crossover is not None:
        ax.axvline(crossover, color="k", ls=":", lw=1)
        ax.annotate(f"crossover\n{crossover:.2f} ms/hop",
                    xy=(crossover, ax.get_ylim()[1] * 0.6),
                    xytext=(crossover * 1.15, ax.get_ylim()[1] * 0.72),
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Per-hop processing delay (ms)")
    ax.set_ylabel("End-to-end delay, Sydney - Tokyo (ms)")
    ax.set_title("Where the MEO backbone starts to pay off")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(outfile, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------
def main():
    print("FAROS - controlled comparison, Sydney to Tokyo")
    print(f"Central angle {SEPARATION_DEG:.2f} deg, "
          f"great-circle {GREAT_CIRCLE_KM:.0f} km")
    print(f"LEO {CONFIG['leo_altitude_km']:.0f} km, "
          f"MEO {CONFIG['meo_altitude_km']:.0f} km, "
          f"LEO-LEO ISL <= {CONFIG['max_isl_leo_leo_km']:.0f} km, "
          f"MEO link <= {CONFIG['max_link_meo_km']:.0f} km, "
          f"elevation mask {CONFIG['min_elevation_deg']:.0f} deg")
    print()

    for pm in (1.0, 5.0):
        print(f"### Per-hop processing delay = {pm} ms")
        sims, rows = run_table(pm)
        print_table(rows, pm)
        print()
        if pm == CONFIG["processing_delay_ms"]:
            for key, fn in [("A  LEO only (40 sats)", "fig6A_leo_only.png"),
                            ("B  LEO 40 + MEO backbone 5", "fig6B_meo_backbone.png")]:
                sim = sims[key]
                _, path = sim.shortest_path("G0", "G1")
                plot_topology(sim, path, key, fn)

    # A third topology figure, in the regime where the backbone IS selected.
    sims_hi, _ = run_table(10.0)
    sim_hi = sims_hi["B  LEO 40 + MEO backbone 5"]
    _, path_hi = sim_hi.shortest_path("G0", "G1")
    plot_topology(sim_hi, path_hi,
                  "B' LEO 40 + MEO backbone, 10 ms/hop processing",
                  "fig6B_backbone_selected.png")

    proc_grid = np.linspace(0.0, 15.0, 301)
    sweep = crossover_sweep(proc_grid)
    xo = find_crossover(sweep)
    print("### Crossover analysis")
    if xo is None:
        print("  No crossover in the 0-15 ms/hop range.")
    else:
        print(f"  The MEO backbone becomes faster than LEO-only when the "
              f"per-hop processing delay exceeds {xo:.2f} ms.")
    plot_crossover(sweep, xo, "fig7_crossover.png")
    print()

    print("### Sensitivity: crossover vs LEO inter-satellite link range")
    hdr = f"{'ISL (km)':>10}{'Hops A':>9}{'Len A (km)':>13}{'Hops B':>9}{'Crossover (ms/hop)':>21}"
    print(hdr); print("-" * len(hdr))
    for r in isl_sensitivity([1000, 1250, 1500, 2000, 2500, 3000, 4000],
                             np.linspace(0.0, 40.0, 401)):
        h_a = "no path" if r["hops_A"] is None else r["hops_A"]
        l_a = "-" if r["length_A_km"] is None else f"{r['length_A_km']:.0f}"
        xo_s = "none < 40" if r["crossover_ms"] is None else f"{r['crossover_ms']:.2f}"
        print(f"{r['isl_km']:>10}{str(h_a):>9}{l_a:>13}{str(r['hops_B']):>9}{xo_s:>21}")

    print("\nFigures written: fig6A_leo_only.png, fig6B_meo_backbone.png, "
          "fig7_crossover.png")


if __name__ == "__main__":
    main()
