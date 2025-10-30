import sys, os
sys.path.append("../../")  # if needed
import pandas as pd
import data_loading as dl
import numpy as np
import matplotlib.pyplot as plt

from numu_tki.cc0pi_analyzer import apply_ccnp0pi_stv

RUN = ["1"]
blinded = True

rundata, mc_weights, data_pot = dl.load_runs(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=True,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=True,
    use_bdt=True,
    load_lee=False,
    load_nue_tki=False,
    load_numu_tki=True,
    blinded=blinded,
    enable_cache=True,
)

import xgboost as xgb
MODEL = "/exp/uboone/app/users/rlalnunt/PELEE_BDT/cc0pi_fstrack_pid_softprob.json"
assert os.path.isfile(MODEL), f"Model file not found: {MODEL}"
print("Using model:", MODEL)
print("xgboost version:", xgb.__version__)

# #print(rundata.keys())
mc = rundata["mc"].copy()
# print(mc)
from numu_tki import cc0pi_analyzer
mc = apply_ccnp0pi_stv(mc, model_path=MODEL)
print(mc)

mcd = mc[np.isfinite(mc["delta_pT"])].copy()

plt.figure()
plt.hist(mcd["delta_pT"], bins=50, range=(0, 1.2), histtype="step", density=False)
plt.xlabel(r"$\Delta p_T$  [GeV/$c$]")
plt.ylabel("Events")
plt.title(rf"Δ$p_T$ (run {','.join(RUN)})")
plt.tight_layout()
plt.savefig("delta_pT.png")
plt.close()

# ---------- robust truth mask for 0pi Np (numu-CC, ≥1 p, 0 π0, 0 charged π) ----------
#import pandas as pd
#import numpy as np

mc = mc.reset_index(drop=True)

# If no weights, set to 1
if "weights" not in mc:
    mc["weights"] = 1.0
w = pd.to_numeric(mc["weights"], errors="coerce").fillna(0.0).astype(float).to_numpy()

def _pick_numeric(df, names, default=None):
    for n in names:
        if n in df.columns:
            s = pd.to_numeric(df[n], errors="coerce")
            return s
    return default

# ccnc (0 = CC)
ccnc = _pick_numeric(mc, ["ccnc", "is_ccnc"])
if ccnc is None:
    raise KeyError("Need a CC/NC truth column (e.g. 'ccnc' or 'is_ccnc').")

# nu PDG (expect 14 for νμ)
nu_pdg = _pick_numeric(mc, ["nu_pdg", "mc_nu_pdg", "truth_nu_pdg"])
if nu_pdg is None:
    raise KeyError("Need a neutrino PDG column (e.g. 'nu_pdg' or 'mc_nu_pdg').")

# number of protons
nproton = _pick_numeric(mc, ["nproton", "n_proton", "nprotons", "n_p"])
if nproton is None:
    raise KeyError("Need a proton multiplicity truth column (e.g. 'nproton').")

# neutral pions (π0); default 0 if unavailable
npi0 = _pick_numeric(mc, ["npi0", "n_pi0", "nneutral_pion", "n_pi0_truth"],
                     default=pd.Series(0, index=mc.index))

# charged pions (π±). Try several options, then try plus/minus split; if nothing, assume 0 and warn.
npi_ch = _pick_numeric(mc, ["npi", "npi_pm", "npi_charged", "n_pi_ch"])
if npi_ch is None:
    npi_plus  = _pick_numeric(mc, ["npi_plus", "n_pi_plus", "npi+"])
    npi_minus = _pick_numeric(mc, ["npi_minus", "n_pi_minus", "npi-"])
    if (npi_plus is not None) and (npi_minus is not None):
        npi_ch = (npi_plus.fillna(0) + npi_minus.fillna(0))
    else:
        # Last resort: assume 0 charged pions if the ntuple doesn’t carry them separately.
        # If that’s too optimistic for your sample, replace with a KeyError instead.
        print("[warn] No charged-pion truth columns found; assuming nπ±=0 for all events.")
        npi_ch = pd.Series(0, index=mc.index)

# Build the truth mask for 0π Np
truth_0piNp = (
    (ccnc.fillna(1) == 0) &          # CC
    (nu_pdg.fillna(0) == 14) &       # νμ
    (npi0.fillna(0) == 0) &          # no π0
    (npi_ch.fillna(0) == 0) &        # no charged π
    (nproton.fillna(0) >= 1)         # ≥1 proton
).to_numpy()

# Selection mask from your new selector (exactly what the code provides)
if "sel_CCNp0pi" not in mc:
    raise KeyError("Expected 'sel_CCNp0pi' from apply_ccnp0pi_stv; not found.")
#sel_mask = mc["sel_CCNp0pi"].astype(bool).to_numpy()
sel_mask = mc.loc[:, mc.columns == "sel_CCNp0pi"].iloc[:, -1].astype(bool).to_numpy()

# Weighted sums (no pandas alignment issues)
tot_sig = float(w[truth_0piNp].sum())                    # total true 0πNp
sel_sig = float(w[truth_0piNp & sel_mask].sum())         # selected true 0πNp
sel_all = float(w[sel_mask].sum())                       # all selected (sig + bkg)

efficiency = (sel_sig / tot_sig) * 100 if tot_sig > 0 else 0.0
purity     = (sel_sig / sel_all) * 100 if sel_all > 0 else 0.0

print(f"[0πNp truth] total={tot_sig:.3f}, selected={sel_sig:.3f}, selected(all)={sel_all:.3f}")
print(f"Efficiency: {efficiency:.2f}%")
print(f"Purity:     {purity:.2f}%")



'''
# --- make indexes & types sane ---
mc = mc.reset_index(drop=True)

# Weights: default to 1.0 if not present
if 'weights' not in mc:
    mc['weights'] = 1.0
w = pd.to_numeric(mc['weights'], errors='coerce').fillna(0.0).astype(float)

# Coerce truth-level columns to numeric (object dtypes can sneak in)
for c in ['ccnc', 'nu_pdg', 'npi0', 'npi', 'nproton']:
    if c in mc:
        mc[c] = pd.to_numeric(mc[c], errors='coerce')

# --- build masks from *this* df and convert to NumPy to avoid alignment ---
sel_mask = mc['sel_CCNp0pi'].astype(bool).to_numpy()

# Define the 0π Np truth for efficiency/purity
# (adjust if your truth definition differs)
need = ['ccnc','nu_pdg','npi0','npi','nproton']
missing = [c for c in need if c not in mc]
if missing:
    raise KeyError(f"Missing truth columns for efficiency/purity: {missing}")

truth_signal = (
    (mc['ccnc'] == 0) &
    (mc['nu_pdg'] == 14) &
    (mc['npi0'] == 0) &
    (mc['npi']  == 0) &
    (mc['nproton'] >= 1)
).to_numpy()

weights = w.to_numpy()

# --- sums (use NumPy indexing, no alignment) ---
tot_sig = float(weights[truth_signal].sum())                  # total true signal
sel_sig = float(weights[truth_signal & sel_mask].sum())       # selected true signal
sel_evt = float(weights[sel_mask].sum())                      # all selected (sig+bkg)

efficiency = (sel_sig / tot_sig) * 100 if tot_sig > 0 else 0.0
purity     = (sel_sig / sel_evt) * 100 if sel_evt > 0 else 0.0

print(f"After cuts: sel_sig={sel_sig:.3f}, sel_evt={sel_evt:.3f}")
print(f"Efficiency: {efficiency:.2f}%")
print(f"Purity:     {purity:.2f}%")
'''

'''
#import numpy as np

# --- 1) Define the truth-level signal: νμ CC with ≥1 proton and 0 pions ---
def _col(df, name, default_value):
#    return df[name] if name in df.columns else default_value

def get_num_col(df, name, default=0):
    """Return a numeric Series aligned to df.index; if column missing, a Series of default."""
#    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

# --- truth mask (signal definition): CC numu with >=1 proton and zero pions
is_cc      = (mc['ccnc'] == 0)
is_numu    = (mc['nu_pdg'] == 14)
has_ge1p   = (mc['nproton'] >= 1)
'''
'''
# zero pions: prefer explicit pion counters if present, else fall back gracefully
npi0    = mc.get('npi0', 0)
npiplus = mc.get('npiplus', 0)
npiminus= mc.get('npiminus', 0)
zero_pi = (npi0.fillna(0) + npiplus.fillna(0) + npiminus.fillna(0) == 0)

truth_signal = (is_cc & is_numu & has_ge1p & zero_pi)
'''
'''
npi0     = get_num_col(mc, 'npi0')
npiplus  = get_num_col(mc, 'npiplus')
npiminus = get_num_col(mc, 'npiminus')
zero_pi  = ((npi0.fillna(0) + npiplus.fillna(0) + npiminus.fillna(0)) == 0)

truth_signal = (is_cc & is_numu & has_ge1p & zero_pi)
# --- selection mask from your analyzer (must be a Series, not a 1-col DataFrame)
sel_mask = mc['sel_CCNp0pi'].fillna(False).astype(bool)

# --- weights (rename if your column is different)
w = mc['weights'].astype(float)

# --- sums
tot_sig   = float(w[truth_signal].sum())            # all true signal before cuts
sel_sig   = float(w[truth_signal & sel_mask].sum()) # selected true signal
sel_all   = float(w[sel_mask].sum())                # all selected (sig + bkg)

efficiency = 100.0 * sel_sig / tot_sig if tot_sig > 0 else 0.0
purity     = 100.0 * sel_sig / sel_all if sel_all > 0 else 0.0

print("Truth mask stats:",
      "CC:", int(is_cc.sum()),
      "numu:", int(is_numu.sum()),
      "has≥1p:", int(has_ge1p.sum()),
      "zeroπ:", int(zero_pi.sum()),
      "signal:", int(truth_signal.sum()))
print(f"Selected events (weighted): total={sel_all:.3f}, signal={sel_sig:.3f}")
print(f"Efficiency: {efficiency:.2f}%")
print(f"Purity:     {purity:.2f}%")
'''

'''
# CC vs NC (prefer 'ccnc'==0 if present)
is_cc   = (_col(mc, "ccnc", None) == 0) if "ccnc" in mc else _col(mc, "is_cc", True)

# νμ (prefer nu_pdg==14 if present)
is_numu = (_col(mc, "nu_pdg", 14) == 14)

# proton count (your printout shows 'nproton' exists)
has_p   = _col(mc, "nproton", 0) >= 1

# zero pions (handle a few common column names; default to 0 if absent)
npi0    = _col(mc, "npi0", 0)
npi     = _col(mc, "npi", None)
npi_pl  = _col(mc, "npi_plus", 0)
npi_mn  = _col(mc, "npi_minus", 0)
npi_ch  = npi if npi is not None else (npi_pl + npi_mn)

zero_pions = (npi0 == 0) & (npi_ch == 0)

truth_signal_0piNp = is_cc & is_numu & has_p & zero_pions

# (Optional) sanity print
print("Truth mask stats:",
      f"CC: {np.sum(is_cc)}",
      f"numu: {np.sum(is_numu)}",
      f"has≥1p: {np.sum(has_p)}",
      f"zeroπ: {np.sum(zero_pions)}",
      f"signal: {np.sum(truth_signal_0piNp)}")

# --- 2) Selection from your new code ---
sel_mask = mc["sel_CCNp0pi"] == True

# --- 3) Weights (fallback to 1.0 if not present) ---
w = mc["weights"] if "weights" in mc.columns else np.ones(len(mc), dtype=float)

# --- 4) Totals, selected, and components ---
tot_sig = float(w[truth_signal_0piNp].sum())                # all true signal before cuts
sel_all = float(w[sel_mask].sum())                          # all selected (sig + bkg)
sel_sig = float(w[sel_mask & truth_signal_0piNp].sum())     # selected true signal
sel_bkg = sel_all - sel_sig

# --- 5) Metrics ---
efficiency = (sel_sig / tot_sig * 100.0) if tot_sig > 0 else 0.0
purity     = (sel_sig / sel_all * 100.0) if sel_all > 0 else 0.0

print("--- CC N p 0π selection (driven by sel_CCNp0pi) ---")
print(f"Total true signal before cuts: {tot_sig:.3f}")
print(f"Selected events (sig+bkg):     {sel_all:.3f}")
print(f"  ↳ selected signal:            {sel_sig:.3f}")
print(f"  ↳ selected background:        {sel_bkg:.3f}")
print(f"Efficiency: {efficiency:.2f}%")
print(f"Purity:     {purity:.2f}%")

'''
