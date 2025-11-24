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
    
    
