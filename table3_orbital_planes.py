"""Monte Carlo over constellation orientation: does more orbital planes help?"""
import math, random, statistics
import orbit_model as T

def rot_matrix(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    return [[cb*cc, -cb*sc, sb],
            [sa*sb*cc+ca*sc, -sa*sb*sc+ca*cc, -sa*cb],
            [-ca*sb*cc+sa*sc, ca*sb*sc+sa*cc, ca*cb]]

def rotate(p, M):
    return tuple(sum(M[i][j]*p[j] for j in range(3)) for i in range(3))

def sample(num_planes, total, rng, proc_ms=1.0):
    T.CFG["total_sats"] = total
    sats = T.build_shell(num_planes, total//num_planes,
                         T.CFG["altitude_km"], T.CFG["inclination_deg"])
    M = rot_matrix(rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi))
    pos = {f"S{i}": rotate(p, M) for i, p in enumerate(sats)}
    pos["G0"] = T.latlon_to_xyz(*T.CFG["sydney"])
    pos["G1"] = T.latlon_to_xyz(*T.CFG["tokyo"])

    import itertools
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in itertools.combinations(pos, 2):
        if a[0] == "G" and b[0] == "G":
            continue
        d = T.dist3(pos[a], pos[b])
        if a[0] == "G" or b[0] == "G":
            g, s = (a, b) if a[0] == "G" else (b, a)
            if T.elevation_deg(pos[g], pos[s]) < T.CFG["min_elevation_deg"]:
                continue
        else:
            if not T.seg_clears_earth(pos[a], pos[b]) or d > T.CFG["max_isl_km"]:
                continue
        graph[a].append((b, d)); graph[b].append((a, d))

    r = T.shortest_path(pos, graph, "G0", "G1", proc_ms)
    k = T.node_disjoint_paths(graph, "G0", "G1")
    return r, k

N = 120
for total in (240, 480):
    print(f"=== {total} satellites, {N} random orientations, Sydney-Tokyo ===")
    print(f"{'planes':>7}{'per plane':>11}{'availability':>14}{'mean delay ms':>15}"
          f"{'mean hops':>11}{'mean disjoint':>15}")
    for np_ in (2, 4, 6, 8, 10, 12, 15, 20):
        if total % np_: continue
        rng = random.Random(20240906 + np_)
        ds, hs, ks, ok = [], [], [], 0
        for _ in range(N):
            r, k = sample(np_, total, rng)
            ks.append(k)
            if r: ok += 1; ds.append(r["total_ms"]); hs.append(r["hops"])
        print(f"{np_:>7}{total//np_:>11}{ok/N*100:>13.0f}%"
              f"{(statistics.mean(ds) if ds else float('nan')):>15.2f}"
              f"{(statistics.mean(hs) if hs else float('nan')):>11.1f}"
              f"{statistics.mean(ks):>15.1f}")
    print()
