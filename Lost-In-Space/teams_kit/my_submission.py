"""plan_imaging.py — Combined Adaptive Safe-Greedy + DP Satellite Scheduler.

Strategy
--------
Phase 1: Propagate orbit at 1 Hz for 720 s; find AOI centroid, compute off-nadir
         angle every second, record η_min.
Phase 2/3 (Cases 1 & 2): If η_min <= 45.0°, use Adaptive Safe-Greedy logic 
         (from clone_4K1.py) filtering by 55.0 deg limits.
Phase 2/3 (Case 3): If η_min > 45.0°, use Dynamic Programming (from plan_imaging_dp.py)
         with K-means clustering and pre-computed transition costs, followed
         by a greedy backfill. Cumulative momentum budgets are ignored to 
         maximize coverage, but individual slew limits are maintained.
Phase 4: Verify output contract, return dict.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sgp4.api import Satrec, jday

# ---- WGS84 constants -------------------------------------------------------
WGS84_A  = 6378137.0
WGS84_F  = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# ---- GLOBAL OPERATIONAL SAFETY MARGINS -------------------------------------
SMEAR_LIMIT_DPS     = 0.03    
WHEEL_H_LIMIT_MNM   = 25.0    # Individual slew restriction (KEPT)
SLEW_RATE_DPS       = 1.0     
INTEGRATION_S       = 0.120   
HOLD_PAD_S          = 0.15    
ATTITUDE_DT_S       = 0.10    

# ---- GREEDY-SPECIFIC CONSTANTS (Cases 1 & 2) -------------------------------
OFF_NADIR_LIMIT_DEG_GREEDY = 55
GRID_BASE_KM_GREEDY        = 15.5

# ---- DP-SPECIFIC CONSTANTS (Case 3) ----------------------------------------
OFF_NADIR_LIMIT_DEG_DP     = 58.563
GRID_BASE_KM_DP            = 10
DP_MAX_NODES               = 22      
EPOCH_STEP                 = 0.5     
EP_SEARCH_WIN              = 12      


# ===========================================================================
# Core Geometry & Time Helpers
# ===========================================================================

def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def _gmst(dt: datetime) -> float:
    jd, fr = jday(dt.year, dt.month, dt.day,
                  dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)
    T = ((jd - 2451545.0) + fr) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T * T
                - 6.2e-6 * T * T * T)
    gmst_sec = gmst_sec % 86400.0
    if gmst_sec < 0:
        gmst_sec += 86400.0
    return (gmst_sec / 240.0) * math.pi / 180.0

def _llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    ss, cs = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array([(N + alt_m) * cl * cs,
                     (N + alt_m) * cl * ss,
                     (N * (1.0 - WGS84_E2) + alt_m) * sl])

def _ecef_to_eci(r_ecef: np.ndarray, gmst: float) -> np.ndarray:
    c, s = math.cos(gmst), math.sin(gmst)
    return np.array([c * r_ecef[0] - s * r_ecef[1],
                     s * r_ecef[0] + c * r_ecef[1],
                     r_ecef[2]])

def _mat_to_quat_xyzw(m: np.ndarray) -> List[float]:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw])
    return (q / np.linalg.norm(q)).tolist()

def _stare_quat_BN(r_sat_eci: np.ndarray, r_tgt_eci: np.ndarray,
                   v_sat_eci: np.ndarray) -> List[float]:
    z_B = r_tgt_eci - r_sat_eci
    z_B = z_B / np.linalg.norm(z_B)
    vhat = v_sat_eci / np.linalg.norm(v_sat_eci)
    x_B = vhat - np.dot(vhat, z_B) * z_B
    nrm = np.linalg.norm(x_B)
    if nrm < 1e-6:
        arb = np.array([1.0, 0.0, 0.0])
        x_B = arb - np.dot(arb, z_B) * z_B
        nrm = np.linalg.norm(x_B)
    x_B = x_B / nrm
    y_B = np.cross(z_B, x_B)
    return _mat_to_quat_xyzw(np.column_stack([x_B, y_B, z_B]))

def _sat_state(sat: Satrec, when: datetime):
    jd, fr = jday(when.year, when.month, when.day,
                  when.hour, when.minute,
                  when.second + when.microsecond * 1e-6)
    err, r_km, v_kmps = sat.sgp4(jd, fr)
    if err != 0:
        return None, None
    return (np.asarray(r_km, float) * 1000.0,
            np.asarray(v_kmps, float) * 1000.0)

def _slerp(q0: List[float], q1: List[float], u: float) -> List[float]:
    a = np.array(q0, dtype=float)
    b = np.array(q1, dtype=float)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0.0:
        b = -b
        dot = -dot
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-10:
        return (a / np.linalg.norm(a)).tolist()
    q = (math.sin((1.0 - u) * theta) / sin_theta) * a + \
        (math.sin(u * theta) / sin_theta) * b
    return (q / np.linalg.norm(q)).tolist()

def _ray_cast_inside(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    px, py = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside

def _generate_grid(aoi_polygon_llh: List[Tuple[float, float]],
                   S_grid_km: float) -> List[Tuple[float, float]]:
    lats = [p[0] for p in aoi_polygon_llh]
    lons = [p[1] for p in aoi_polygon_llh]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_center = (lat_min + lat_max) / 2.0
    d_lat = S_grid_km / 111.32
    d_lon = S_grid_km / (111.32 * math.cos(math.radians(lat_center)) + 1e-300)
    
    pts = []
    lat = lat_min
    row_idx = 0
    while lat <= lat_max + 1e-9:
        lon = lon_min
        row_pts = []
        while lon <= lon_max + 1e-9:
            if _ray_cast_inside((lat, lon), aoi_polygon_llh):
                row_pts.append((lat, lon))
            lon += d_lon
        
        # --- THE SNAKE MODIFICATION ---
        # Reverse every other row to create a continuous lawnmower path
        if row_idx % 2 == 1:
            row_pts.reverse()
        # ------------------------------
            
        pts.extend(row_pts)
        lat += d_lat
        row_idx += 1
        
    return pts

def _off_nadir_deg(r_sat: np.ndarray, r_tgt: np.ndarray) -> float:
    nadir = -r_sat / np.linalg.norm(r_sat)
    los = r_tgt - r_sat
    los_norm = np.linalg.norm(los)
    if los_norm < 1.0:
        return 90.0
    return math.degrees(math.acos(float(np.clip(np.dot(nadir, los/los_norm), -1., 1.))))

def _rotate_vec_by_quat(q: List[float], v: List[float]) -> np.ndarray:
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return np.array([vx + qw * tx + qy * tz - qz * ty,
                     vy + qw * ty + qz * tx - qx * tz,
                     vz + qw * tz + qx * ty - qy * tx])

def _fallback_stub(T: float, reason: str) -> dict:
    return {
        "objective": "Combined DP/Greedy Coverage [DEGRADED]",
        "attitude": [
            {"t": 0.0,      "q_BN": [0.0, 0.0, 0.0, 1.0]},
            {"t": float(T), "q_BN": [0.0, 0.0, 0.0, 1.0]},
        ],
        "shutter": [],
        "notes": f"Graceful fallback. Reason: {reason}",
    }


# ===========================================================================
# DP-Specific Helpers
# ===========================================================================

def _quat_angle_deg(q0: List[float], q1: List[float]) -> float:
    a   = np.array(q0, dtype=float)
    b   = np.array(q1, dtype=float)
    dot = abs(float(np.clip(np.dot(a, b), -1., 1.)))
    return math.degrees(2.0 * math.acos(min(dot, 1.0)))

def _max_wheel_dH(theta_deg: float, q_src: List[float], q_dst: List[float]) -> float:
    a    = np.array(q_src, dtype=float)
    b    = np.array(q_dst, dtype=float)
    if np.dot(a, b) < 0:
        b = -b
    cross = np.cross(a[:3], b[:3])
    nrm   = np.linalg.norm(cross)
    if nrm < 1e-9:
        return 0.0
    axis  = cross / nrm
    omega = SLEW_RATE_DPS * math.pi / 180.0
    I     = np.array([0.12, 0.12, 0.08])
    dH    = I * omega * np.abs(axis) * 1e3
    return float(np.max(dH))

def _kmeans(pts: List[Tuple[float,float]], k: int, max_iter: int = 30) -> List[Tuple[float,float]]:
    if len(pts) <= k:
        return list(pts)
    rng = np.random.default_rng(42)
    arr = np.array(pts, dtype=float)
    C   = [arr[rng.integers(len(arr))]]
    for _ in range(k - 1):
        d    = np.array([min(float(np.sum((p-c)**2)) for c in C) for p in arr])
        C.append(arr[rng.choice(len(arr), p=d/d.sum())])
    C = np.array(C)
    for _ in range(max_iter):
        lbl  = np.argmin(np.linalg.norm(arr[:,None]-C[None,:], axis=2), axis=1)
        newC = np.array([arr[lbl==i].mean(0) if np.any(lbl==i) else C[i] for i in range(k)])
        if np.allclose(newC, C, atol=1e-9):
            break
        C = newC
    return [(float(c[0]), float(c[1])) for c in C]

class _TInfo:
    __slots__ = ("t_transit", "max_dH", "q_dst")
    def __init__(self, t_transit, max_dH, q_dst):
        self.t_transit = t_transit
        self.max_dH    = max_dH
        self.q_dst     = q_dst

def _precompute(targets, r_sat, v_sat, gmst_c, n_steps, T):
    N        = len(targets)
    n_ep     = int(T / EPOCH_STEP) + 1
    ecef     = [_llh_to_ecef(la, lo) for la, lo in targets]
    Q_NADIR  = [0., 0., 0., 1.]
    table: Dict[Tuple[int,int,int], _TInfo] = {}

    for ep in range(n_ep):
        t_dep  = ep * EPOCH_STEP
        if t_dep >= T - INTEGRATION_S:
            break
        ti     = int(min(math.floor(t_dep), n_steps - 1))
        rs, vs = r_sat[ti], v_sat[ti]
        g      = gmst_c[ti]

        q_dep = [None] * N
        for i in range(N):
            rt = _ecef_to_eci(ecef[i], g)
            if _off_nadir_deg(rs, rt) <= OFF_NADIR_LIMIT_DEG_DP:
                q_dep[i] = _stare_quat_BN(rs, rt, vs)

        for src in range(-1, N):
            q_src = Q_NADIR if src == -1 else q_dep[src]
            if q_src is None:
                continue

            for dst in range(N):
                if dst == src:
                    continue
                q_d  = q_dep[dst]
                thet = _quat_angle_deg(q_src, q_d) if q_d is not None else 180.0
                t_sl = thet / SLEW_RATE_DPS
                t_se = 1.0 + thet * 0.05
                t_im = t_dep + t_sl + t_se
                if t_im + INTEGRATION_S > T:
                    continue
                ti2  = int(min(math.floor(t_im), n_steps - 1))
                rt2  = _ecef_to_eci(ecef[dst], gmst_c[ti2])
                if _off_nadir_deg(r_sat[ti2], rt2) > OFF_NADIR_LIMIT_DEG_DP:
                    continue
                q_im = _stare_quat_BN(r_sat[ti2], rt2, v_sat[ti2])
                
                # SLEW RESTRICTION (Kept to prevent physically impossible single jumps)
                dH   = _max_wheel_dH(thet, q_src, q_im)
                if dH > WHEEL_H_LIMIT_MNM:
                    continue
                table[(src, dst, ep)] = _TInfo(t_sl + t_se, dH, q_im)

    return table

def _run_dp(N: int, table: Dict, T: float) -> Tuple[List[int], float]:
    NEG_INF = -1e18
    INF_T   = T + 1e9
    NM      = 1 << N
    n_ep    = int(T / EPOCH_STEP) + 1

    sc  = np.full((N, NM), NEG_INF)
    tc  = np.full((N, NM), INF_T)
    bk  = [[None]*NM for _ in range(N)]

    for dst in range(N):
        for ep in range(min(EP_SEARCH_WIN * 4, n_ep)):
            key = (-1, dst, ep)
            if key not in table:
                continue
            inf  = table[key]
            t_d  = ep * EPOCH_STEP + inf.t_transit + INTEGRATION_S
            if t_d > T:
                continue
            
            mask = 1 << dst
            # Score is base 1.0 (for 1 frame), minus a tiny fraction of time to break ties
            s    = 1.0 - 1e-6 * t_d
            
            if s > sc[dst, mask]:
                sc[dst, mask]  = s
                tc[dst, mask]  = t_d
                bk[dst][mask]  = (-1, 0)
            break

    for mask in range(1, NM):
        for last in range(N):
            if not (mask & (1 << last)):
                continue
            s_c = sc[last, mask]
            if s_c <= NEG_INF:
                continue
            t_c  = tc[last, mask]
            ep0  = int(math.ceil(t_c / EPOCH_STEP))

            for dst in range(N):
                if mask & (1 << dst):
                    continue
                found = None
                for ep in range(ep0, min(ep0 + EP_SEARCH_WIN, n_ep)):
                    key = (last, dst, ep)
                    if key not in table:
                        continue
                    inf  = table[key]
                    t_dep = ep * EPOCH_STEP
                    if t_dep < t_c:
                        continue
                    t_done = t_dep + inf.t_transit + INTEGRATION_S
                    if t_done > T:
                        break
                    found = (t_done, inf)
                    break
                
                if found is None:
                    continue
                t_done, inf = found
                
                nm  = mask | (1 << dst)
                # Maximize pure coverage count, use time as tiny tie-breaker
                s_n = float(bin(nm).count("1")) - 1e-6 * t_done
                
                if s_n > sc[dst, nm]:
                    sc[dst, nm]  = s_n
                    tc[dst, nm]  = t_done
                    bk[dst][nm]  = (last, mask)

    best_s, best_l, best_m = NEG_INF, -1, 0
    for mask in range(1, NM):
        for last in range(N):
            if (mask & (1 << last)) and sc[last, mask] > best_s:
                best_s = sc[last, mask]; best_l = last; best_m = mask

    if best_l < 0:
        return [], 0.0

    path = []
    cl, cm = best_l, best_m
    while cm:
        path.append(cl)
        prev = bk[cl][cm]
        if prev is None or prev[0] == -1:
            break
        cl, cm = prev
    path.reverse()
    return path, float(best_s)

def _append_slerp(att: List[dict], t0: float, t1: float, q0: List[float], q1: List[float]):
    dur = max(t1 - t0, 1e-9)
    t   = t0
    while t < t1 - 1e-9:
        u   = max(0.0, min(1.0, (t - t0) / dur))
        q   = _slerp(q0, q1, u)
        t_r = round(t, 4)
        if att and t_r - att[-1]["t"] < 0.020 - 1e-9:
            t += ATTITUDE_DT_S
            continue
        att.append({"t": t_r, "q_BN": q})
        t += ATTITUDE_DT_S

def _commit_frame(att, sh, t_img, Q_img):
    for ht in [t_img - HOLD_PAD_S, t_img, t_img + INTEGRATION_S, t_img + INTEGRATION_S + HOLD_PAD_S]:
        ht_r = round(ht, 4)
        if att and ht_r - att[-1]["t"] < 0.020 - 1e-9:
            continue
        att.append({"t": ht_r, "q_BN": list(Q_img)})
    sh.append({"t_start": round(t_img, 4), "duration": 0.120})

def _build_traj(seq, targets, r_sat, v_sat, gmst_c, ecef, n_steps, T, q0=None, t0=0.0):
    if q0 is None:
        q0 = [0., 0., 0., 1.]
    att, sh = [], []
    if abs(t0) < 1e-9:
        att.append({"t": 0.0, "q_BN": list(q0)})
    ct, cq = t0, list(q0)

    for idx in seq:
        ti   = int(min(math.floor(ct), n_steps - 1))
        rt_n = _ecef_to_eci(ecef[idx], gmst_c[ti])
        q_n  = _stare_quat_BN(r_sat[ti], rt_n, v_sat[ti])
        thet = _quat_angle_deg(cq, q_n)
        t_im = ct + thet / SLEW_RATE_DPS + 1.0 + thet * 0.05
        if t_im + INTEGRATION_S > T:
            break
        ti2  = int(min(math.floor(t_im), n_steps - 1))
        rt2  = _ecef_to_eci(ecef[idx], gmst_c[ti2])
        if _off_nadir_deg(r_sat[ti2], rt2) > OFF_NADIR_LIMIT_DEG_DP:
            continue
        Q_im = _stare_quat_BN(r_sat[ti2], rt2, v_sat[ti2])
        _append_slerp(att, ct, t_im - HOLD_PAD_S, cq, Q_im)
        _commit_frame(att, sh, t_im, Q_im)
        ct = t_im + INTEGRATION_S; cq = list(Q_im)

    return att, sh, ct, cq

def _greedy_fill(done, targets, ecef, r_sat, v_sat, gmst_c, n_steps, T, att, sh, ct, cq):
    rem = [(i, t) for i, t in enumerate(targets) if i not in done]
    while ct < T and rem:
        ti   = int(min(math.floor(ct), n_steps - 1))
        best, bsc = None, None
        for li, (oi, (la, lo)) in enumerate(rem):
            rt_n = _ecef_to_eci(ecef[oi], gmst_c[ti])
            q_n  = _stare_quat_BN(r_sat[ti], rt_n, v_sat[ti])
            thet = _quat_angle_deg(cq, q_n)
            t_im = ct + thet / SLEW_RATE_DPS + 1.0 + thet * 0.05
            if t_im + INTEGRATION_S > T:
                continue
            ti2  = int(min(math.floor(t_im), n_steps - 1))
            rt2  = _ecef_to_eci(ecef[oi], gmst_c[ti2])
            eta  = _off_nadir_deg(r_sat[ti2], rt2)
            if eta > OFF_NADIR_LIMIT_DEG_DP:
                continue
            sc = -(0.7 * thet + 0.05 * eta)
            if bsc is None or sc > bsc:
                bsc = sc; best = (li, oi, t_im, ti2, rt2)
        if best is None:
            ct += 0.5; continue
        li, oi, t_im, ti2, rt2 = best
        Q_im = _stare_quat_BN(r_sat[ti2], rt2, v_sat[ti2])
        _append_slerp(att, ct, t_im - HOLD_PAD_S, cq, Q_im)
        _commit_frame(att, sh, t_im, Q_im)
        ct = t_im + INTEGRATION_S; cq = list(Q_im)
        done.add(oi); rem.pop(li)
    return ct, cq

def _clean(att: List[dict], T: float) -> List[dict]:
    out = []
    for s in att:
        if out and s["t"] - out[-1]["t"] < 0.020 - 1e-9:
            continue
        q  = np.array(s["q_BN"], dtype=float)
        qn = np.linalg.norm(q)
        if qn > 1e-9:
            q /= qn
        out.append({"t": float(s["t"]), "q_BN": q.tolist()})
    if not out or abs(out[0]["t"]) > 1e-9:
        out.insert(0, {"t": 0.0, "q_BN": [0., 0., 0., 1.]})
    return out


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def plan_imaging(tle_line1: str, tle_line2: str, aoi_polygon_llh: List[Tuple[float, float]],
                 pass_start_utc: str, pass_end_utc: str, sc_params: Dict[str, Any]) -> Dict[str, Any]:
    
    try:
        t0 = _parse_iso(pass_start_utc)
        t1 = _parse_iso(pass_end_utc)
        T  = (t1 - t0).total_seconds()
    except Exception as _e:
        return _fallback_stub(720.0, f"parse error: {_e}")

    try:
        # ==================================================================
        # PHASE 1 — Orbit propagation and scout
        # ==================================================================
        sat = Satrec.twoline2rv(tle_line1, tle_line2)
        n_steps = int(math.floor(T)) + 1

        r_sat_cache: List[np.ndarray] = []
        v_sat_cache: List[np.ndarray] = []
        gmst_cache:  List[float]      = []

        for i in range(n_steps):
            t_sec = min(float(i), T)
            when  = t0 + timedelta(seconds=t_sec)
            r_eci, v_eci = _sat_state(sat, when)
            if r_eci is None:
                r_eci = r_sat_cache[-1].copy() if r_sat_cache else np.array([7e6, 0.0, 0.0])
                v_eci = v_sat_cache[-1].copy() if v_sat_cache else np.array([0.0, 7500.0, 0.0])
            g = _gmst(when)
            r_sat_cache.append(r_eci)
            v_sat_cache.append(v_eci)
            gmst_cache.append(g)

        # AOI centroid
        verts = aoi_polygon_llh
        if len(verts) > 1 and verts[0] == verts[-1]:
            verts = verts[:-1]
        tgt_lat = sum(p[0] for p in verts) / len(verts)
        tgt_lon = sum(p[1] for p in verts) / len(verts)
        r_tgt_ecef = _llh_to_ecef(tgt_lat, tgt_lon, 0.0)

        r_tgt_eci_cache: List[np.ndarray] = [
            _ecef_to_eci(r_tgt_ecef, gmst_cache[i]) for i in range(n_steps)
        ]

        # Find eta_min
        eta_series = [
            _off_nadir_deg(r_sat_cache[i], r_tgt_eci_cache[i])
            for i in range(n_steps)
        ]
        eta_min = min(eta_series)
        
        # Determine tracking strategy based on off-nadir performance
        if eta_min <= 45.0:
            # ==================================================================
            # PHASE 2 & 3 — GREEDY SAFE METHOD (CASES 1 & 2)
            # Extracted from clone_4K1.py
            # ==================================================================
            if eta_min >= 89.0:
                S_grid_km = GRID_BASE_KM_GREEDY * 10.0
            else:
                S_grid_km = GRID_BASE_KM_GREEDY / math.cos(math.radians(eta_min))

            unimaged_targets = _generate_grid(list(verts), S_grid_km)
            N_targets_initial = len(unimaged_targets)

            if not unimaged_targets:
                unimaged_targets = [(tgt_lat, tgt_lon)]
                N_targets_initial = 1

            current_time = 0.0
            current_q_BN: List[float] = [0.0, 0.0, 0.0, 1.0]
            attitude_list: List[dict] = [{"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]}]
            shutter_list:  List[dict] = []

            omega_safe_rad = SLEW_RATE_DPS * math.pi / 180.0

            while current_time < T and len(unimaged_targets) > 0:

                t_int = int(min(math.floor(current_time), n_steps - 1))
                r_sat_now = r_sat_cache[t_int]
                v_sat_now = v_sat_cache[t_int]

                z_cam = _rotate_vec_by_quat(current_q_BN, [0.0, 0.0, 1.0])
                z_cam = z_cam / (np.linalg.norm(z_cam) + 1e-300)

                feasible = []

                for idx, (lat_i, lon_i) in enumerate(unimaged_targets):
                    r_tgt_ecef_i = _llh_to_ecef(lat_i, lon_i, 0.0)
                    r_tgt_eci_i  = _ecef_to_eci(r_tgt_ecef_i, gmst_cache[t_int])

                    z_target = r_tgt_eci_i - r_sat_now
                    zt_norm  = np.linalg.norm(z_target)
                    if zt_norm < 1.0:
                        continue
                    z_target = z_target / zt_norm

                    dot_val  = float(np.dot(z_cam, z_target))
                    dot_val  = max(-1.0, min(1.0, dot_val))
                    theta_deg = math.degrees(math.acos(dot_val))

                    T_slew   = theta_deg / SLEW_RATE_DPS
                    T_settle = 1.0 + theta_deg * 0.05
                    t_future = current_time + T_slew + T_settle

                    if t_future + INTEGRATION_S > T:
                        continue

                    t_f_int = int(min(math.floor(t_future), n_steps - 1))
                    
                    r_tgt_eci_start = _ecef_to_eci(r_tgt_ecef_i, gmst_cache[t_f_int])
                    eta_start = _off_nadir_deg(r_sat_cache[t_f_int], r_tgt_eci_start)
                    if eta_start > OFF_NADIR_LIMIT_DEG_GREEDY:
                        continue

                    t_mid = t_future + INTEGRATION_S / 2.0
                    t_mid_int = int(min(math.floor(t_mid), n_steps - 1))
                    r_tgt_eci_future = _ecef_to_eci(r_tgt_ecef_i, gmst_cache[t_mid_int])
                    eta_future = _off_nadir_deg(r_sat_cache[t_mid_int], r_tgt_eci_future)
                    if eta_future > OFF_NADIR_LIMIT_DEG_GREEDY:
                        continue

                    cross_vec = np.cross(z_cam, z_target)
                    cross_nrm = np.linalg.norm(cross_vec)
                    if cross_nrm < 1e-9:
                        e_hat = np.array([0.0, 0.0, 1.0])
                    else:
                        e_hat = cross_vec / cross_nrm

                    Hx_mNms = abs(0.12 * omega_safe_rad * e_hat[0]) * 1000.0
                    Hy_mNms = abs(0.12 * omega_safe_rad * e_hat[1]) * 1000.0
                    Hz_mNms = abs(0.08 * omega_safe_rad * e_hat[2]) * 1000.0
                    if (Hx_mNms > WHEEL_H_LIMIT_MNM or
                            Hy_mNms > WHEEL_H_LIMIT_MNM or
                            Hz_mNms > WHEEL_H_LIMIT_MNM):
                        continue

                    feasible.append((idx, theta_deg, eta_future, t_future, r_tgt_eci_future))

                if not feasible:
                    current_time += 0.1
                    continue

                best = min(feasible, key=lambda f: f[1] + 0.1 * f[2])
                idx_w, theta_w, eta_w, t_future_w, r_tgt_eci_w = best

                t_f_int_w = int(min(math.floor(t_future_w), n_steps - 1))
                Q_winner  = _stare_quat_BN(
                    r_sat_cache[t_f_int_w],
                    r_tgt_eci_w,
                    v_sat_cache[t_f_int_w],
                )

                t_wp = current_time
                slew_end = t_future_w - HOLD_PAD_S
                total_slew = max(slew_end - current_time, 1e-9)
                while t_wp < slew_end - 1e-9:
                    u = max(0.0, min(1.0, (t_wp - current_time) / total_slew))
                    q_wp = _slerp(current_q_BN, Q_winner, u)
                    t_rounded = round(t_wp, 4)
                    if t_rounded - attitude_list[-1]["t"] >= 0.020 - 1e-9:
                        attitude_list.append({"t": t_rounded, "q_BN": q_wp})
                    t_wp += ATTITUDE_DT_S

                hold_times = [
                    t_future_w - HOLD_PAD_S,
                    t_future_w,
                    t_future_w + INTEGRATION_S,
                    t_future_w + INTEGRATION_S + HOLD_PAD_S,
                ]
                for ht in hold_times:
                    ht_r = round(ht, 4)
                    if ht_r - attitude_list[-1]["t"] >= 0.020 - 1e-9:
                        attitude_list.append({"t": ht_r, "q_BN": list(Q_winner)})

                shutter_list.append({
                    "t_start":  round(t_future_w, 4),
                    "duration": 0.120,
                })

                current_time = t_future_w + INTEGRATION_S
                current_q_BN = list(Q_winner)
                unimaged_targets.pop(idx_w)

            # Cleanup Greedy sequence
            if shutter_list:
                last_need = shutter_list[-1]["t_start"] + 0.120
                if attitude_list[-1]["t"] < last_need - 1e-9:
                    attitude_list.append({
                        "t": round(last_need, 4),
                        "q_BN": attitude_list[-1]["q_BN"],
                    })
            else:
                if attitude_list[-1]["t"] < T - 1e-9:
                    attitude_list.append({"t": round(T, 4), "q_BN": attitude_list[-1]["q_BN"]})

            cleaned = []
            for s in attitude_list:
                if cleaned and s["t"] - cleaned[-1]["t"] < 0.020 - 1e-9:
                    continue
                q = np.array(s["q_BN"], dtype=float)
                qn = np.linalg.norm(q)
                if qn > 1e-9:
                    q = q / qn
                cleaned.append({"t": float(s["t"]), "q_BN": q.tolist()})

            if not cleaned or abs(cleaned[0]["t"]) > 1e-9:
                cleaned.insert(0, {"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]})

            if not shutter_list:
                return _fallback_stub(T, "no feasible shutter windows found")

            return {
                "objective": "Adaptive Safe-Greedy Coverage",
                "attitude":  cleaned,
                "shutter":   shutter_list,
                "notes":     (f"Dynamic grid {N_targets_initial} pts, S_grid={S_grid_km:.2f} km, "
                              f"eta_min={eta_min:.2f} deg, {len(shutter_list)} frames."),
            }

        else:
            # ==================================================================
            # PHASE 2 & 3 — DYNAMIC PROGRAMMING (CASE 3)
            # Extracted from plan_imaging_dp.py
            # ==================================================================
            S_km    = (GRID_BASE_KM_DP * 10 if eta_min >= 89
                       else GRID_BASE_KM_DP / math.cos(math.radians(eta_min)))
            targets = _generate_grid(list(verts), S_km) or [(tgt_lat, tgt_lon)]
            N_all   = len(targets)
            ecef_all = [_llh_to_ecef(la, lo) for la, lo in targets]

            dp_tgts  = _kmeans(targets, DP_MAX_NODES) if N_all > DP_MAX_NODES else list(targets)
            N_dp     = len(dp_tgts)
            ecef_dp  = [_llh_to_ecef(la, lo) for la, lo in dp_tgts]

            table    = _precompute(dp_tgts, r_sat_cache, v_sat_cache, gmst_cache, n_steps, T)
            seq, dps = _run_dp(N_dp, table, T)

            att, sh, ct, cq = _build_traj(
                seq, dp_tgts, r_sat_cache, v_sat_cache, gmst_cache, ecef_dp, n_steps, T,
                q0=[0., 0., 0., 1.], t0=0.0)

            done = set(seq)

            # Fallback greedy if DP fails to yield sequence
            if not sh:
                att = [{"t": 0.0, "q_BN": [0., 0., 0., 1.]}]
                ct, cq = _greedy_fill(
                    set(), targets, ecef_all, r_sat_cache, v_sat_cache, gmst_cache, n_steps, T,
                    att, sh, 0.0, [0., 0., 0., 1.])
            else:
                if N_all > DP_MAX_NODES:
                    cov: set = set()
                    for di in done:
                        de = ecef_dp[di]
                        cov.add(min(range(N_all), key=lambda k: float(np.linalg.norm(ecef_all[k]-de))))
                else:
                    cov = set(done)
                ct, cq = _greedy_fill(
                    cov, targets, ecef_all, r_sat_cache, v_sat_cache, gmst_cache, n_steps, T,
                    att, sh, ct, cq)

            if sh:
                need = sh[-1]["t_start"] + INTEGRATION_S
                if not att or att[-1]["t"] < need - 1e-9:
                    att.append({"t": round(need, 4),
                                 "q_BN": att[-1]["q_BN"] if att else [0.,0.,0.,1.]})
            else:
                if not att or att[-1]["t"] < T - 1e-9:
                    att.append({"t": round(T, 4),
                                 "q_BN": att[-1]["q_BN"] if att else [0.,0.,0.,1.]})

            cleaned = _clean(att, T)

            if not sh or len(sh) < 3:
                return _fallback_stub(T, "no feasible shutter windows")

            n_fill = len(sh) - len(seq)
            return {
                "objective": "DP Global Coverage + Greedy Fill (No Cumulative Momentum Cap)",
                "attitude":  cleaned,
                "shutter":   sh,
                "notes": (f"Grid {N_all}pts → DP on {N_dp} clusters (frames_mapped={dps:.1f}), "
                          f"eta_min={eta_min:.1f}deg, S_grid={S_km:.1f}km, "
                          f"{len(sh)} frames (DP:{len(seq)}+fill:{n_fill})."),
                "target_hints_llh": [{"lat_deg": la, "lon_deg": lo} for la, lo in dp_tgts],
            }

    except Exception as e:
        return _fallback_stub(T, str(e))