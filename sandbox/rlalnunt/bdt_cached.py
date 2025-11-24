import sys, os
sys.path.append("../../")  # if needed
import pandas as pd
import data_loading as dl
import numpy as np
import matplotlib.pyplot as plt

from cc0pi_cached import load_runs_with_cc0pi_bdt

rundata, mc_weights, data_pot = load_runs_with_cc0pi_bdt(
    ["1"],
    model_path="/exp/uboone/app/users/rlalnunt/PELEE_BDT/cc0pi_fstrack_pid_softprob.json",
    data="bnb",
    loadshowervariables=False,
    loadnumuvariables=True,
    load_numu_tki=False,
    loadpi0variables=False,
    loadrecoveryvars=False,
    loadsystematics=True,
    blinded=True,
    enable_cache=True,   # turns on the HDF cache
    overwrite=False,      # don’t overwrite cached result
    use_bdt=False
)
mc = rundata["mc"]

# print(mc)

# print(mc.filter(regex="^xgb_|cc0pi", axis=1).head())  # BDT cols / selection flags, if any
# print(len(mc), "rows in MC after BDT pass-through")

def _b(x):  # safe bool
    return x.fillna(False).astype(bool)

def compute_eff_pur_df(df: pd.DataFrame) -> pd.DataFrame:
    # selection
    stages = [
        ("no cuts",                                       lambda d: d["is_mc"]==d["is_mc"]),  # always True
        ("in FV",                                         lambda d: _b(d["sel_reco_vertex_in_FV"])),
        ("starts contained",                              lambda d: _b(d["sel_reco_vertex_in_FV"]) & _b(d["sel_pfp_starts_in_PCV"])),
        ("CCincl",                                        lambda d: _b(d["sel_nu_mu_cc"])),
        ("#mu momentum limits",                           lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_muon_passed_mom_cuts"])),
        ("no showers",                                    lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_no_reco_showers"]) & _b(d["sel_muon_passed_mom_cuts"])),
        ("#mu contained",                                 lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_no_reco_showers"]) & _b(d["sel_muon_passed_mom_cuts"]) & _b(d["sel_muon_contained"])),
        ("#mu quality",                                   lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_no_reco_showers"]) & _b(d["sel_muon_passed_mom_cuts"]) & _b(d["sel_muon_contained"]) & _b(d["sel_muon_quality_ok"])),
        ("has p candidate",                               lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_no_reco_showers"]) & _b(d["sel_muon_passed_mom_cuts"]) & _b(d["sel_muon_contained"]) & _b(d["sel_muon_quality_ok"]) & _b(d["sel_has_p_candidate"])),
        ("p contained",                                   lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_no_reco_showers"]) & _b(d["sel_muon_passed_mom_cuts"]) & _b(d["sel_muon_contained"]) & _b(d["sel_muon_quality_ok"]) & _b(d["sel_has_p_candidate"]) & _b(d["sel_protons_contained"])),
        ("proton PID",                                    lambda d: _b(d["sel_nu_mu_cc"]) & _b(d["sel_no_reco_showers"]) & _b(d["sel_muon_passed_mom_cuts"]) & _b(d["sel_muon_contained"]) & _b(d["sel_muon_quality_ok"]) & _b(d["sel_has_p_candidate"]) & _b(d["sel_protons_contained"]) & _b(d["sel_passed_proton_pid_cut"])),
        ("p momentum limits (== sel_CC1p0pi)",            lambda d: _b(d["sel_CC1p0pi"])),
    ]

    mc  = _b(df["is_mc"])
    sig = _b(df["mc_is_signal"]) & mc  # C++ uses signal = "mc_is_signal && is_mc"

    rows = []
    for idx, (name, sel_fun) in enumerate(stages, start=1):
        sel = sel_fun(df) & mc                      # selection && is_mc
        num_signal           = int(sig.sum())       # denominator for efficiency
        num_selected         = int(sel.sum())       # denominator for purity
        num_selected_signal  = int((sig & sel).sum())

        eff = (num_selected_signal / num_signal)   if num_signal   else np.nan
        pur = (num_selected_signal / num_selected) if num_selected else np.nan

        rows.append(dict(
            step=idx, stage=name,
            num_signal=num_signal,
            num_selected=num_selected,
            num_selected_signal=num_selected_signal,
            efficiency=eff, purity=pur,
        ))

    return pd.DataFrame(rows)

# Example:
results = compute_eff_pur_df(mc)  # df contains your selection booleans from apply_ccnp0pi_stv(...)
print(results)
