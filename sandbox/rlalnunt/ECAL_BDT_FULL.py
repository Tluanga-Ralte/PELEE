import sys, os
sys.path.append("../../")  # if needed
import pandas as pd
import data_loading as dl
import numpy as np
import matplotlib.pyplot as plt
from numu_tki.cc0pi_analyzer import apply_ccnp0pi_stv
from cc0pi_cached import load_runs_with_cc0pi_bdt

RUN = ["1"]
blinded = True

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
#print(mc)
print("mc_is_cc1p0pi_signal" in rundata["mc"].columns)
#print(mc["mc_num_protons"].values)
print(len(mc["mc_is_cc1p0pi_signal"]))


'''
rundata, mc_weights, data_pot = dl.load_runs_with_cc0pi_bdt(
    RUN,
    data="bnb",
    loadpi0variables=False,
    loadshowervariables=False,
    loadrecoveryvars=False,
    loadsystematics=True,
    numupresel=False,
    loadnumuvariables=True,
    use_bdt=False,
    load_lee=False,
    load_nue_tki=False,
    load_numu_tki=False,
    blinded=blinded,
    enable_cache=True,
)

import xgboost as xgb
MODEL = "/exp/uboone/app/users/rlalnunt/PELEE_BDT/cc0pi_fstrack_pid_softprob.json"
assert os.path.isfile(MODEL), f"Model file not found: {MODEL}"
print("Using model:", MODEL)
print("xgboost version:", xgb.__version__)
'''
