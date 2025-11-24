import sys, os
sys.path.append("../../")  # if needed
import pandas as pd
import data_loading as dl
import numpy as np
import matplotlib.pyplot as plt
from cc0pi_cached import _apply_cc0pi_cached, _df_fp, _hash_file, _code_fp, generate_hash


model_path = "/exp/uboone/app/users/rlalnunt/PELEE_BDT/cc0pi_fstrack_pid_softprob.json"

#Load the run normally (no BDT yet)
rundata, weights, data_pot = dl.load_runs(
    ["1"],
    enable_cache=True,
    data="bnb",
    loadshowervariables=False,
    loadnumuvariables=True,
    load_numu_tki=False,
    loadpi0variables=False,
    loadrecoveryvars=False,
    loadsystematics=True,
    blinded=True,
    use_bdt=False,
)

mc_full = rundata["mc"]

mc_small = mc_full.head(100).copy()

print(mc_small)
 
