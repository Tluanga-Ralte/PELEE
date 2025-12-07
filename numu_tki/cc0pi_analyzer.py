# cc0pi_analyzer.py
# Analysis macro for CCNp0pi/STV with C++-aligned selection and MC-truth logic.
# Re-uses selection developed by Steven Gardiner (22 Apr 2023)
# Author: Ralte Lalnuntluanga (28 Sep 2025) — Python port with MC truth

from __future__ import annotations
import math
import numpy as np
import pandas as pd

# Optional: xgboost (for the PID booster)
try:
    import xgboost as xgb
except Exception:
    xgb = None

# ===== constants =====
LOW_FLOAT = -1e30
BOGUS_INDEX = -1

# PDG
ELECTRON_NEUTRINO = 12
MUON = 13
MUON_NEUTRINO = 14
TAU_NEUTRINO = 16
PROTON = 2212
NEUTRON = 2112
PI_ZERO = 111
PI_PLUS = 211

# masses (GeV)
TARGET_MASS = 37.215526
NEUTRON_MASS = 0.93956541
PROTON_MASS = 0.93827208
MUON_MASS = 0.10565837
PI_PLUS_MASS = 0.13957000
BINDING_ENERGY = 0.02478

# cuts
DEFAULT_PROTON_PID_CUT = 0.2
LEAD_P_MIN_MOM_CUT = 0.250
LEAD_P_MAX_MOM_CUT = 1.000
MUON_P_MIN_MOM_CUT = 0.100
MUON_P_MIN_WC_MOM_CUT = 0.121
MUON_P_MAX_MOM_CUT = 2.000
CHARGED_PI_MOM_CUT = 0.07
CHARGED_PI_WC_MOM_CUT = 0.161
MUON_MOM_QUALITY_CUT = 0.25

TOPO_SCORE_CUT = 0.15
COSMIC_IP_CUT = 25.0
TRACK_SCORE_CUT = 0.5

# ===== FV (for vertex + BDT "trk_contained" @ track END) =====
FV_X_MIN, FV_X_MAX = 21.5, 234.85
FV_Y_MIN, FV_Y_MAX = -95.0, 95.0
FV_Z_MIN, FV_Z_MAX = 21.5, 966.8

# ===== PCV (PFParticle STARTS, muon/proton track ENDS) =====
PCV_X_MIN, PCV_X_MAX = 10.0, 246.35
PCV_Y_MIN, PCV_Y_MAX = -106.5, 106.5
PCV_Z_MIN, PCV_Z_MAX = 10.0, 1026.8

# ===== EventCategory (match EventCategory.hh) =====
kUnknown      = 0
kSignalCCQE   = 1
kSignalCCMEC  = 2
kSignalCCRES  = 3
kSignalOther  = 4
kNuMuCCNpi    = 5
kNuMuCC0pi0p  = 6   # (unused in current C++ selection but kept for parity)
kNuMuCCOther  = 7
kNuECC        = 8
kNC           = 9
kOOFV         = 10
kOther        = 11

CHARGED_CURRENT = 0
NEUTRAL_CURRENT = 1

# ===== helpers =====
def real_sqrt(x: float) -> float:
    return 0.0 if x < 0.0 else math.sqrt(x)

def in_FV(x: float, y: float, z: float) -> bool:
    return (FV_X_MIN < x < FV_X_MAX and
            FV_Y_MIN < y < FV_Y_MAX and
            FV_Z_MIN < z < FV_Z_MAX)

def in_proton_containment_vol(x: float, y: float, z: float) -> bool:
    return (PCV_X_MIN < x < PCV_X_MAX and
            PCV_Y_MIN < y < PCV_Y_MAX and
            PCV_Z_MIN < z < PCV_Z_MAX)

def is_meson_or_antimeson(pdg: int) -> bool:
    abs_pdg = abs(pdg)
    if abs_pdg >= 9900000: return False
    if (abs_pdg // 1000) % 10 != 0: return False    # thousands digit
    if (abs_pdg // 100) % 10 == 0: return False     # hundreds digit
    if 901 <= abs_pdg <= 930: return False          # PDFs
    if abs_pdg in (110, 990, 998, 999, 100): return False
    return True


try:
    import awkward as ak
except Exception:
    ak = None

def _force_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (str, bytes, bytearray)):
        return [v]  # treat strings as scalars, not char lists
    if np is not None and isinstance(v, np.ndarray):
        return v.tolist()
    if ak is not None and getattr(ak, "is_any_array", None) and ak.is_any_array(v):
        return ak.to_list(v)
    # fallback: wrap scalar or try to list() if it’s a finite sequence
    try:
        return list(v)
    except Exception:
        return [v]



def _pick_first(row, names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default

# ===== STV computations (same formulas as C++) =====
def compute_stvs(p3mu: np.ndarray, p3p: np.ndarray):
    # p3mu, p3p arrays length-3
    delta_pT_vec = p3mu[:2] + p3p[:2]
    delta_pT = float(np.linalg.norm(delta_pT_vec))

    # delta_phiT
    den = (np.linalg.norm(p3mu[:2]) * np.linalg.norm(p3p[:2]))
    delta_phiT = float(np.arccos(np.clip((-(p3mu[0]*p3p[0] + p3mu[1]*p3p[1]))/den, -1.0, 1.0))) if den > 0 else np.nan

    # delta_alphaT
    den2 = (np.linalg.norm(p3mu[:2]) * np.linalg.norm(delta_pT_vec))
#     delta_alphaT = float(np.arccos(np.clip((-(p3mu[0]*delta_pT_vec[0] - 0 + -p3mu[1]*delta_pT_vec[1]))/den2, -1.0, 1.0))) if den2 > 0 else np.nan
    num2 = -(p3mu[0]*delta_pT_vec[0] + p3mu[1]*delta_pT_vec[1])
    delta_alphaT = float(np.arccos(np.clip(num2/den2, -1.0, 1.0))) if den2 > 0 else np.nan

    # (same as C++, minus algebraic reshuffle: -p_mu dot deltapT)

    Emu = math.sqrt(MUON_MASS**2 + float(np.dot(p3mu, p3mu)))
    Ep  = math.sqrt(PROTON_MASS**2 + float(np.dot(p3p,  p3p)))
    R = TARGET_MASS + p3mu[2] + p3p[2] - Emu - Ep
    
    Ecal = Emu + (Ep - PROTON_MASS) + BINDING_ENERGY 

    mf = TARGET_MASS - NEUTRON_MASS + BINDING_ENERGY
    delta_pL = 0.5*R - (mf*mf + delta_pT*delta_pT)/(2.0*R) if R != 0 else np.nan
    pn = math.sqrt((delta_pL if np.isfinite(delta_pL) else 0.0)**2 + delta_pT**2)

    # transverse components (arXiv:1910.08658)
    zhat = np.array([0.0, 0.0, 1.0])
    xT = np.cross(zhat, p3mu)[:2]
    xT = xT/np.linalg.norm(xT) if np.linalg.norm(xT) > 0 else np.array([np.nan, np.nan])

    yT = (-p3mu[:2])
    yT = yT/np.linalg.norm(yT) if np.linalg.norm(yT) > 0 else np.array([np.nan, np.nan])

    delta_pTx = float(np.dot(xT, delta_pT_vec)) if np.all(np.isfinite(xT)) else np.nan
    delta_pTy = float(np.dot(yT, delta_pT_vec)) if np.all(np.isfinite(yT)) else np.nan

    # opening angle
    nmu = np.linalg.norm(p3mu); np_ = np.linalg.norm(p3p)
    theta_mu_p = float(np.arccos(np.clip(np.dot(p3mu, p3p)/(nmu*np_), -1.0, 1.0))) if (nmu > 0 and np_ > 0) else np.nan

    return dict(
        delta_pT=delta_pT,
        delta_phiT=delta_phiT,
        delta_alphaT=delta_alphaT,
        delta_pL=delta_pL,
        pn=pn,
        delta_pTx=delta_pTx,
        delta_pTy=delta_pTy,
        theta_mu_p=theta_mu_p,
        Ecal=Ecal,
    )

# ===== Booster I/O =====
def _load_booster(model_path: str):
    if xgb is None:
        raise RuntimeError("xgboost not installed. `pip install xgboost`")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

# ===== BDT classification (same 7 features, same class order: 0=Other,1=Mu,2=Pi,3=P) =====
def classify_tracks_for_event(booster, row):
    # these are already lists thanks to one-time normalization
    gen        = row["pfp_generation_v"]
    tscore     = row["trk_score_v"]
    trk_dist   = row["trk_distance_v"]
    pid_score  = row["trk_llr_pid_score_v"]
    chi2_p     = row.get("trk_pid_chipr_v", [])
    KE_p       = row["trk_energy_proton_v"]
    endx       = row["trk_sce_end_x_v"]
    endy       = row["trk_sce_end_y_v"]
    endz       = row["trk_sce_end_z_v"]
    n_trk_dau  = row.get("pfp_trk_daughters_v", [])
    n_shr_dau  = row.get("pfp_shr_daughters_v", [])
    mom_range  = row["trk_range_muon_mom_v"]
    mom_mcs    = row["trk_mcs_muon_mom_v"]

    n = len(gen)
    xgb_pid_vec   = [-1]*n
    xgb_score_vec = [[] for _ in range(n)]

    trk_end_contained = [in_FV(endx[i], endy[i], endz[i]) for i in range(n)]

    if n_trk_dau and n_shr_dau and len(n_trk_dau)==n and len(n_shr_dau)==n:
        ndaughters = [int(n_trk_dau[i]) + int(n_shr_dau[i]) for i in range(n)]
    else:
        ndaughters = [0]*n

    def _safe(x, default=0.0):
        try:
            xf = float(x)
            return xf if np.isfinite(xf) else default
        except Exception:
            return default

    feats, indices = [], []
    for i in range(n):
        if int(gen[i]) != 2:
            continue
        if float(tscore[i]) <= TRACK_SCORE_CUT:
            continue

        prange = float(mom_range[i]) if i < len(mom_range) else np.nan
        pmcs   = float(mom_mcs[i]) if i < len(mom_mcs) else np.nan
        rel    = (pmcs - prange)/prange if (np.isfinite(prange) and prange>0) else 0.0

        fs = [
            _safe(trk_dist[i], 0.0),
            _safe(pid_score[i], 0.0),
            _safe(chi2_p[i] if i < len(chi2_p) else 0.0, 0.0),
            _safe(KE_p[i], 0.0),
            1.0 if trk_end_contained[i] else 0.0,
            float(ndaughters[i]),
            _safe(rel, 0.0),
        ]
        if not all(np.isfinite(val) for val in fs):
            continue
        feats.append(fs); indices.append(i)

    if feats:
        dmat = xgb.DMatrix(np.asarray(feats, dtype=np.float32))
        probs = booster.predict(dmat)
        for idx, prob in zip(indices, probs):
            k = int(np.argmax(prob))
            xgb_pid_vec[idx] = k
            xgb_score_vec[idx] = prob.tolist()

    counts = {0:0, 1:0, 2:0, 3:0, -1:0}
    for k in xgb_pid_vec:
        counts[k] = counts.get(k, 0) + 1
    return xgb_pid_vec, xgb_score_vec, counts


# ===== reco selection helpers =====
def _pick_muon_candidate(xgb_pid_vec, xgb_score_vec):
    mu_indices = [i for i,k in enumerate(xgb_pid_vec) if k == 1]
    if not mu_indices:
        return BOGUS_INDEX
    if len(mu_indices) == 1:
        return mu_indices[0]
    best_i, best_s = BOGUS_INDEX, LOW_FLOAT
    for i in mu_indices:
        s = xgb_score_vec[i][1] if xgb_score_vec[i] else LOW_FLOAT
        if s > best_s:
            best_s, best_i = s, i
    return best_i

def _muon_momentum(row, idx, is_contained):
    rng = float(row["trk_range_muon_mom_v"][idx])
    mcs = float(row["trk_mcs_muon_mom_v"][idx])
    if is_contained:
        # flipped-track veto if available
        if ("trk_bragg_mu_fwd_preferred_v" in row) and ("trk_pid_chimu_v" in row):
            try:
                fwd = int(_force_list(row["trk_bragg_mu_fwd_preferred_v"])[idx])
                ntracks = int(row.get("n_tracks", len(_force_list(row["trk_len_v"]))))
                chi2_mu = float(_force_list(row["trk_pid_chimu_v"])[idx])
                if ntracks == 1 and fwd == 0 and chi2_mu > 6.0:
                    return LOW_FLOAT
            except Exception:
                pass
        return rng
    if mcs < 0.11:
        return LOW_FLOAT
    return mcs - 0.0361*mcs + 0.04

def _first_proton_window_and_contained(row, idx) -> tuple[bool,bool,float]:
    KEp = float(row["trk_energy_proton_v"][idx])
    p_mom = real_sqrt(KEp*KEp + 2.0*PROTON_MASS*KEp)
    in_window = (LEAD_P_MIN_MOM_CUT <= p_mom <= LEAD_P_MAX_MOM_CUT)
    endx = float(row["trk_sce_end_x_v"][idx])
    endy = float(row["trk_sce_end_y_v"][idx])
    endz = float(row["trk_sce_end_z_v"][idx])
    contained = in_proton_containment_vol(endx, endy, endz)
    return in_window, contained, p_mom

def _find_leading_proton(row, xgb_pid_vec, mu_idx):
    n = len(xgb_pid_vec)
    lead_idx, lead_len = BOGUS_INDEX, LOW_FLOAT
    for i in range(n):
        if i == mu_idx: continue
        if xgb_pid_vec[i] != 3:  # class P
            continue
        try:
            trk_len = float(row["trk_len_v"][i])
            if not np.isfinite(trk_len) or trk_len <= 0:
                continue
        except Exception:
            continue
        in_window, contained, _ = _first_proton_window_and_contained(row, i)
        if contained and in_window and trk_len > lead_len:
            lead_len = trk_len
            lead_idx = i
    return lead_idx

def _vector_from_dir_and_p(dirx, diry, dirz, p):
    v = np.array([dirx, diry, dirz], dtype=float)
    n = np.linalg.norm(v)
    if n == 0 or not np.isfinite(n) or p <= 0:
        return np.array([LOW_FLOAT, LOW_FLOAT, LOW_FLOAT], dtype=float)
    return (v / n) * p

# ===== MC truth categorization (mirror C++ categorize_event) =====
def _categorize_mc(row):
    # defaults
    out = dict(
        is_mc=False,
        mc_neutrino_is_numu=False,
        mc_vertex_in_FV=False,
        mc_muon_in_mom_range=False,
        mc_muon_in_wc_mom_range=False,
        mc_lead_p_in_mom_range=False,
        mc_no_fs_pi0=True,
        mc_no_charged_pi_above_threshold=True,
        mc_no_charged_pi_above_wc_threshold=True,
        mc_no_fs_mesons=True,
        mc_is_signal=False,
        mc_is_cc0pi_signal=False,
        mc_is_cc1p0pi_signal=False,
        mc_is_cc0pi_wc_signal=False,
        mc_num_protons=0,
        mc_num_protons_in_window=0,
        mc_num_neutrons=0,
        mc_num_charged_pions=0,
        mc_num_wc_charged_pions=0,
        category=kUnknown,
    )

    # --- neutrino PDG ---
    mc_nu_pdg = 0
    for name in ("mc_nu_pdg", "nu_pdg", "truth_nu_pdg"):
        if name in row:
            val = row[name]
            try:
                mc_nu_pdg = int(val)
                break
            except Exception:
                pass

    abs_pdg = abs(mc_nu_pdg)
    is_mc = abs_pdg in (ELECTRON_NEUTRINO, MUON_NEUTRINO, TAU_NEUTRINO)
    out["is_mc"] = is_mc
    if not is_mc:
        return out  # data event

    # --- truth vertex ---
    vx = vy = vz = np.nan
    if "mc_nu_vtx_x" in row and row["mc_nu_vtx_x"] is not None:
        vx = float(row["mc_nu_vtx_x"])
        vy = float(row["mc_nu_vtx_y"])
        vz = float(row["mc_nu_vtx_z"])
    elif "true_nu_vtx_x" in row and row["true_nu_vtx_x"] is not None:
        vx = float(row["true_nu_vtx_x"])
        vy = float(row["true_nu_vtx_y"])
        vz = float(row["true_nu_vtx_z"])

    out["mc_vertex_in_FV"] = in_FV(vx, vy, vz)

    # --- CC vs NC ---
    ccnc = NEUTRAL_CURRENT
    for name in ("mc_ccnc", "ccnc"):
        if name in row and row[name] is not None:
            try:
                ccnc = int(row[name])
            except Exception:
                ccnc = NEUTRAL_CURRENT
            break

    out["mc_neutrino_is_numu"] = (mc_nu_pdg == MUON_NEUTRINO)

    # early categories (these match the C++ structure)
    if not out["mc_vertex_in_FV"]:
        out["category"] = kOOFV
        return out
    elif ccnc == NEUTRAL_CURRENT:
        out["category"] = kNC
        return out
    elif not out["mc_neutrino_is_numu"]:
        if mc_nu_pdg == ELECTRON_NEUTRINO and ccnc == CHARGED_CURRENT:
            out["category"] = kNuECC
        else:
            out["category"] = kOther
        return out

    # --- daughters (truth final state) ---
    pdgs = _force_list(row.get("mc_pdg", []))
    E    = _force_list(row.get("mc_E",   []))
    px   = _force_list(row.get("mc_px",  []))
    py   = _force_list(row.get("mc_py",  []))
    pz   = _force_list(row.get("mc_pz",  []))

    lead_p_mom = LOW_FLOAT

    for i, pdg_val in enumerate(pdgs):
        try:
            pdg = int(pdg_val)
        except Exception:
            continue

        # meson flag
        if is_meson_or_antimeson(pdg):
            out["mc_no_fs_mesons"] = False

        # guard against too-short E list
        Ej = None
        if i < len(E):
            try:
                Ej = float(E[i])
            except Exception:
                Ej = None

        # muon
        if pdg == MUON and Ej is not None:
            mom = real_sqrt(Ej*Ej - MUON_MASS*MUON_MASS)
            if MUON_P_MIN_MOM_CUT <= mom <= MUON_P_MAX_MOM_CUT:
                out["mc_muon_in_mom_range"] = True
            if MUON_P_MIN_WC_MOM_CUT <= mom <= MUON_P_MAX_MOM_CUT:
                out["mc_muon_in_wc_mom_range"] = True

        # proton
        elif pdg == PROTON and Ej is not None:
            mom = real_sqrt(Ej*Ej - PROTON_MASS*PROTON_MASS)
            if mom > lead_p_mom:
                lead_p_mom = mom
            out["mc_num_protons"] += 1
            if LEAD_P_MIN_MOM_CUT <= mom <= LEAD_P_MAX_MOM_CUT:
                out["mc_num_protons_in_window"] += 1

        # neutron
        elif pdg == NEUTRON:
            out["mc_num_neutrons"] += 1

        # charged pion
        elif abs(pdg) == PI_PLUS and Ej is not None:
            out["mc_num_charged_pions"] += 1
            mom = real_sqrt(Ej*Ej - PI_PLUS_MASS*PI_PLUS_MASS)
            if mom > CHARGED_PI_MOM_CUT:
                out["mc_no_charged_pi_above_threshold"] = False
            if mom > CHARGED_PI_WC_MOM_CUT:
                out["mc_no_charged_pi_above_wc_threshold"] = False
                out["mc_num_wc_charged_pions"] += 1

        # pi0
        elif pdg == PI_ZERO:
            out["mc_no_fs_pi0"] = False

    # leading proton window
    if LEAD_P_MIN_MOM_CUT <= lead_p_mom <= LEAD_P_MAX_MOM_CUT:
        out["mc_lead_p_in_mom_range"] = True

    # --- signal flags (mirror C++ definitions) ---
    out["mc_is_signal"] = (
        out["mc_vertex_in_FV"] and
        out["mc_neutrino_is_numu"] and
        out["mc_muon_in_mom_range"] and
        out["mc_lead_p_in_mom_range"] and
        out["mc_no_fs_mesons"]
    )

    out["mc_is_cc0pi_signal"] = (
        out["mc_vertex_in_FV"] and
        out["mc_neutrino_is_numu"] and
        out["mc_muon_in_mom_range"] and
        out["mc_no_fs_pi0"] and
        out["mc_no_charged_pi_above_threshold"]
    )
    
    out["mc_is_cc1p0pi_signal"] = (
        out["mc_is_cc0pi_signal"] and
        out["mc_num_protons_in_window"] == 1
    )

    out["mc_is_cc0pi_wc_signal"] = (
        out["mc_vertex_in_FV"] and
        out["mc_neutrino_is_numu"] and
        out["mc_muon_in_wc_mom_range"] and
        out["mc_no_fs_pi0"] and
        out["mc_no_charged_pi_above_wc_threshold"]
    )

    # --- category by interaction mode for cc0pi signal ---
    interaction = -1
    for name in ("mc_interaction", "interaction"):
        if name in row and row[name] is not None:
            try:
                interaction = int(row[name])
            except Exception:
                interaction = -1
            break

    if out["mc_is_cc0pi_signal"]:
        if interaction == 0:
            out["category"] = kSignalCCQE
        elif interaction == 10:
            out["category"] = kSignalCCMEC
        elif interaction == 1:
            out["category"] = kSignalCCRES
        else:
            out["category"] = kSignalOther
    elif (not out["mc_no_fs_pi0"]) or (not out["mc_no_charged_pi_above_threshold"]):
        out["category"] = kNuMuCCNpi
    else:
        out["category"] = kNuMuCCOther

    return out


# ===== per-event main =====
def analyze_event(row, booster):
    # ensure list-like branches are lists
#     for k in [
#         "pfp_generation_v","trk_score_v","trk_distance_v","trk_len_v",
#         "trk_llr_pid_score_v","trk_pid_chipr_v","trk_energy_proton_v",
#         "trk_dir_x_v","trk_dir_y_v","trk_dir_z_v",
#         "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
#         "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
#         "trk_range_muon_mom_v","trk_mcs_muon_mom_v",
#         "pfp_trk_daughters_v","pfp_shr_daughters_v",
#         "trk_bragg_mu_fwd_preferred_v","trk_pid_chimu_v",
       
#         "mc_pdg","mc_E","mc_px","mc_py","mc_pz",
#     ]:
#         if k in row:
#             row[k] = _force_list(row[k])

    # --- MC truth block (categorize + truth STVs) ---
    mc = _categorize_mc(row)

    # truth STVs (only if we have a CC muon and a leading proton)
    mc_stv = {f"mc_{k}": np.nan for k in
              ["delta_pT","delta_phiT","delta_alphaT","delta_pL","pn","delta_pTx","delta_pTy","theta_mu_p","Ecal"]}
    if mc["is_mc"]:
        pdgs = row.get("mc_pdg", []) or []
        E    = row.get("mc_E", []) or []
        px   = row.get("mc_px",  []) or []
        py   = row.get("mc_py",  []) or []
        pz   = row.get("mc_pz",  []) or []
     
        mu_idx = next((i for i,p in enumerate(pdgs) if int(p) == MUON), None)

        lead_idx = None; lead_mag2 = LOW_FLOAT
        for i,p in enumerate(pdgs):
            if int(p) == PROTON and i < len(px) and i < len(py) and i < len(pz):
                mag2 = float(px[i])**2 + float(py[i])**2 + float(pz[i])**2
                if mag2 > lead_mag2:
                    lead_mag2 = mag2; lead_idx = i

        if mu_idx is not None and lead_idx is not None:
            mc_p3_mu = np.array([float(px[mu_idx]), float(py[mu_idx]), float(pz[mu_idx])], dtype=float)
            mc_p3_p  = np.array([float(px[lead_idx]), float(py[lead_idx]), float(pz[lead_idx])], dtype=float)
            _stv = compute_stvs(mc_p3_mu, mc_p3_p)
            mc_stv.update({f"mc_{k}": v for k,v in _stv.items()})

    # --- reco “no showers” ---
    gen = row["pfp_generation_v"]; tscore = row["trk_score_v"]
    reco_shower_count = sum(1 for i in range(len(gen)) if int(gen[i])==2 and float(tscore[i])<=TRACK_SCORE_CUT)
    sel_no_reco_showers = (reco_shower_count == 0)

    # --- numu CC preselection bits
    sel_reco_vertex_in_FV = in_FV(
        float(row["reco_nu_vtx_sce_x"]),
        float(row["reco_nu_vtx_sce_y"]),
        float(row["reco_nu_vtx_sce_z"])
    )
    sel_topo_cut_passed = float(row["topological_score"]) > TOPO_SCORE_CUT
    sel_cosmic_ip_cut_passed = float(row["CosmicIP"]) > COSMIC_IP_CUT

    # PFParticle starts in PCV (gen==2)
    sx = row["trk_sce_start_x_v"]; sy = row["trk_sce_start_y_v"]; sz = row["trk_sce_start_z_v"]
    sel_pfp_starts_in_PCV = True
    for i in range(len(gen)):
        if int(gen[i]) != 2: continue
        sel_pfp_starts_in_PCV &= in_proton_containment_vol(float(sx[i]), float(sy[i]), float(sz[i]))

    nslice = int(row.get("nslice", 1))
    sel_presel = (nslice == 1 and sel_reco_vertex_in_FV and sel_pfp_starts_in_PCV and sel_topo_cut_passed and sel_no_reco_showers)

    # --- BDT classify (only if presel)
    if sel_presel:
        xgb_pid_vec, xgb_score_vec, counts = classify_tracks_for_event(booster, row)
    else:
        xgb_pid_vec, xgb_score_vec = ([-1]*len(gen), [[] for _ in gen])
        counts = {k:0 for k in [0,1,2,3,-1]}

    # --- muon candidate + momentum
    mu_idx = _pick_muon_candidate(xgb_pid_vec, xgb_score_vec)
    sel_has_muon_candidate = (mu_idx != BOGUS_INDEX)

    muon_contained = False
    muon_passed_mom_cuts = False
    muon_passed_wc_mom_cuts = False
    muon_quality_ok = False
    p3mu = np.array([LOW_FLOAT,LOW_FLOAT,LOW_FLOAT], dtype=float)

    if sel_has_muon_candidate:
        ex = float(row["trk_sce_end_x_v"][mu_idx]); ey = float(row["trk_sce_end_y_v"][mu_idx]); ez = float(row["trk_sce_end_z_v"][mu_idx])
        muon_contained = in_proton_containment_vol(ex, ey, ez)

        mu_p = _muon_momentum(row, mu_idx, muon_contained)
        if not np.isfinite(mu_p) or mu_p <= 0:
            mu_p = LOW_FLOAT

        # quality (|range - mcs| / range < cut)
        rng = float(row["trk_range_muon_mom_v"][mu_idx])
        mcs = float(row["trk_mcs_muon_mom_v"][mu_idx])
        muon_quality_ok = (rng > 0 and abs(rng - mcs)/rng < MUON_MOM_QUALITY_CUT)

        if MUON_P_MIN_MOM_CUT <= mu_p <= MUON_P_MAX_MOM_CUT:
            muon_passed_mom_cuts = True
        if MUON_P_MIN_WC_MOM_CUT <= mu_p <= MUON_P_MAX_MOM_CUT:
            muon_passed_wc_mom_cuts = True

        p3mu = _vector_from_dir_and_p(float(row["trk_dir_x_v"][mu_idx]),
                                      float(row["trk_dir_y_v"][mu_idx]),
                                      float(row["trk_dir_z_v"][mu_idx]),
                                      mu_p)

        # special case (not contained & cosθ < −0.9) -> invalidate
        if (not muon_contained) and np.isfinite(p3mu[2]):
            norm = np.linalg.norm(p3mu)
            costh = p3mu[2]/(norm if norm>0 else 1.0)
            if costh < -0.9:
                p3mu[:] = LOW_FLOAT

    sel_nu_mu_cc = sel_presel and sel_has_muon_candidate

    # --- proton candidates & leading proton
    has_p_candidate = False
    protons_contained = False
    passed_proton_pid_cut = False
    lead_p_idx = _find_leading_proton(row, xgb_pid_vec, mu_idx)

    num_p_candidates = 0
    for i,k in enumerate(xgb_pid_vec):
        if i == mu_idx: continue
        if k != 3: continue
        in_win, contained, _ = _first_proton_window_and_contained(row, i)
        if in_win and contained:
            has_p_candidate = True
            protons_contained = True
            passed_proton_pid_cut = True
            num_p_candidates += 1

    lead_p_passed_mom_cuts = False
    p3p = np.array([LOW_FLOAT,LOW_FLOAT,LOW_FLOAT], dtype=float)
    if lead_p_idx != BOGUS_INDEX:
        in_win, contained, p_mom = _first_proton_window_and_contained(row, lead_p_idx)
        lead_p_passed_mom_cuts = in_win
        p3p = _vector_from_dir_and_p(float(row["trk_dir_x_v"][lead_p_idx]),
                                     float(row["trk_dir_y_v"][lead_p_idx]),
                                     float(row["trk_dir_z_v"][lead_p_idx]),
                                     p_mom)

    # --- final decisions
    sel_CCNp0pi = (sel_nu_mu_cc and sel_no_reco_showers and
                   muon_passed_mom_cuts and muon_contained and muon_quality_ok and
                   has_p_candidate and protons_contained and
                   lead_p_passed_mom_cuts)

    sel_CC1p0pi = (sel_CCNp0pi and num_p_candidates == 1)

    sel_CC0pi = (sel_nu_mu_cc and sel_no_reco_showers and
                 muon_passed_mom_cuts and
                 (counts.get(2,0) == 0) and (counts.get(0,0) == 0) and (counts.get(-1,0) == 0))

    sel_CC0pi_wc = (sel_nu_mu_cc and sel_no_reco_showers and
                    muon_passed_wc_mom_cuts and
                    (counts.get(2,0) == 0) and (counts.get(0,0) == 0) and (counts.get(-1,0) == 0))

    # --- reco STVs (need both vectors)
    have_mu = np.all(np.isfinite(p3mu)) and (p3mu[0] != LOW_FLOAT)
    have_p  = np.all(np.isfinite(p3p))  and (p3p[0]  != LOW_FLOAT)
    stv = compute_stvs(p3mu, p3p) if (have_mu and have_p) else {k: np.nan for k in
        ["delta_pT","delta_phiT","delta_alphaT","delta_pL","pn","delta_pTx","delta_pTy","theta_mu_p","Ecal"]}

    # --- pack output
    out = dict(
        # reco selection
        sel_presel=sel_presel,
        sel_nu_mu_cc=sel_nu_mu_cc,
        sel_reco_vertex_in_FV=sel_reco_vertex_in_FV,
        sel_topo_cut_passed=sel_topo_cut_passed,
        sel_cosmic_ip_cut_passed=sel_cosmic_ip_cut_passed,
        sel_pfp_starts_in_PCV=sel_pfp_starts_in_PCV,
        sel_no_reco_showers=sel_no_reco_showers,
        sel_has_muon_candidate=sel_has_muon_candidate,
        sel_muon_contained=muon_contained,
        sel_muon_quality_ok=muon_quality_ok,
        sel_muon_passed_mom_cuts=muon_passed_mom_cuts,
        sel_muon_passed_wc_mom_cuts=muon_passed_wc_mom_cuts,
        sel_has_p_candidate=has_p_candidate,
        sel_passed_proton_pid_cut=passed_proton_pid_cut,
        sel_protons_contained=protons_contained,
        sel_lead_p_passed_mom_cuts=lead_p_passed_mom_cuts,
        sel_CCNp0pi=sel_CCNp0pi,
        sel_CC1p0pi=sel_CC1p0pi,
        sel_CC0pi=sel_CC0pi,
        sel_CC0pi_wc=sel_CC0pi_wc,

        # indices
        muon_candidate_idx=mu_idx,
        lead_p_candidate_idx=lead_p_idx,

        # BDT counts
        sel_n_bdt_other=counts.get(0,0),
        sel_n_bdt_muon=counts.get(1,0),
        sel_n_bdt_pion=counts.get(2,0),
        sel_n_bdt_proton=counts.get(3,0),
        sel_n_bdt_invalid=counts.get(-1,0),

        # reco vectors
        p3_mu_x=p3mu[0], p3_mu_y=p3mu[1], p3_mu_z=p3mu[2],
        p3_lead_p_x=p3p[0], p3_lead_p_y=p3p[1], p3_lead_p_z=p3p[2],

        # reco STVs
        **stv,

        # raw BDT outputs (lists)
        xgb_pid_vec=xgb_pid_vec,
        xgb_score_vec=xgb_score_vec,

        # MC truth flags + category
        **mc,

        # MC truth STVs
        **mc_stv,
    )
    return out

# ===== public API =====
# def apply_ccnp0pi_stv(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
#     """
#     Adds CCNp0pi selection booleans, candidate indices, BDT outputs, reco p3 vectors,
#     reco STVs, AND MC-truth category/flags + truth STVs (if MC branches present).
#     Expects PeLEE-like branches as list-like columns. Returns a new DataFrame.
#     """
#     booster = _load_booster(model_path)

#     required = [
#         "pfp_generation_v","trk_score_v","trk_distance_v","trk_len_v",
#         "trk_llr_pid_score_v","trk_energy_proton_v",
#         "trk_dir_x_v","trk_dir_y_v","trk_dir_z_v",
#         "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
#         "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
#         "trk_range_muon_mom_v","trk_mcs_muon_mom_v",
#         "topological_score","CosmicIP",
#         "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
#         "nslice",
#     ]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise KeyError(f"Missing required columns: {missing}")

#     # MC truth arrays are optional; if present, they’ll be used automatically:
#     # mc_pdg, mc_E, mc_px, mc_py, mc_pz, mc_nu_pdg/nu_pdg/truth_nu_pdg,
#     # mc_ccnc/ccnc, mc_interaction/interaction, mc_nu_vtx_{x,y,z}/true_nu_vtx_{x,y,z}

#     out = df.apply(lambda r: analyze_event(r, booster), axis=1, result_type="expand")
#     return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
def apply_ccnp0pi_stv(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    booster = _load_booster(model_path)

    required = [
        "pfp_generation_v","trk_score_v","trk_distance_v","trk_len_v",
        "trk_llr_pid_score_v","trk_energy_proton_v",
        "trk_dir_x_v","trk_dir_y_v","trk_dir_z_v",
        "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
        "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
        "trk_range_muon_mom_v","trk_mcs_muon_mom_v",
        "topological_score","CosmicIP",
        "reco_nu_vtx_sce_x","reco_nu_vtx_sce_y","reco_nu_vtx_sce_z",
        "nslice",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # ---- ONE-TIME normalization of vector-like columns (reco + MC) ----
    vector_cols = [
        # reco
        "pfp_generation_v","trk_score_v","trk_distance_v","trk_len_v",
        "trk_llr_pid_score_v","trk_pid_chipr_v","trk_energy_proton_v",
        "trk_dir_x_v","trk_dir_y_v","trk_dir_z_v",
        "trk_sce_end_x_v","trk_sce_end_y_v","trk_sce_end_z_v",
        "trk_sce_start_x_v","trk_sce_start_y_v","trk_sce_start_z_v",
        "trk_range_muon_mom_v","trk_mcs_muon_mom_v",
        "pfp_trk_daughters_v","pfp_shr_daughters_v",
        "trk_bragg_mu_fwd_preferred_v","trk_pid_chimu_v",
        # MC (optional)
        "mc_pdg","mc_E","mc_px","mc_py","mc_pz",
    ]
    present = [c for c in vector_cols if c in df.columns]
    for c in present:
        df[c] = df[c].map(_force_list)  # <- normalize once

    # no more _force_list calls inside analyze_event / classifier
    out = df.apply(lambda r: analyze_event(r, booster), axis=1, result_type="expand")
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

