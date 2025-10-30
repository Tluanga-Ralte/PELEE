import sys, os 
sys.path.append("../../")
from data_loading import load_runs

RUN = ["1"]  # this can be a list of several runs, i.e. [1,2,3]
blinded = True

#For 1mu1p/1e1p -> For full sample 
rundata, mc_weights, data_pot = load_runs(
    RUN,
    data="bnb",  # which data to load
    # truth_filtered_sets=["nue", "drt", "nc_pi0", "cc_pi0", "cc_nopi", "cc_cpi", "nc_nopi", "nc_cpi"],
    # Which truth-filtered MC sets to load in addition to the main MC set. At least nu_e and dirt
    # are highly recommended because the statistics at the final level of the selection are very low.
    truth_filtered_sets=["nue", "drt"],
    # Choose which additional variables to load. Which ones are required may depend on the selection
    # you wish to apply.
    loadpi0variables=True,
    loadshowervariables=True,
    loadrecoveryvars=True,
    loadsystematics=True,
    loadnumuvariables=True, # True for 1mu1p/ False for 1e1p
    numupresel=False,
    # Load the nu_e set one more time with the LEE weights applied
    load_lee=False,
    # With the cache enabled, by default the loaded dataframes will be stored as HDF5 files
    # in the 'cached_dataframes' folder. This will speed up subsequent loading of the same data.
    enable_cache=True,
    # Since this is Open Data, we are allowed to unblind the data. By default, the data is blinded.
    blinded=blinded,
    load_numu_tki=True,
    load_nue_tki=True
)


#For 1mu1p
RUN = ["3"]
rundata_mu, mc_weights_mu, data_pot_mu = load_runs(
    RUN,
    data="opendata_bnb",  # which data to load opendata for now
    truth_filtered_sets=["nue", "drt"],
    loadpi0variables=True,
    loadshowervariables=True,
    loadrecoveryvars=True,
    loadsystematics=True,
    loadnumuvariables=True, # True for 1mu1p/ False for 1e1p
    numupresel=False, # Set to False for full sample
    load_lee=False,
    enable_cache=True,
    # Since this is Open Data, we are allowed to unblind the data. By default, the data is blinded.
    blinded=blinded,
    load_numu_tki=True,
    load_nue_tki=True
)

# #1mu1p - NP -> numu presel=off
# rundata_np, mc_weights_np, data_pot_np = load_runs(
#     RUN,
#     data="bnb",  
#     truth_filtered_sets=["nue", "drt"],
#     # Choose which additional variables to load. Which ones are required may depend on the selection
#     # you wish to apply.
#     loadpi0variables=True,
#     loadshowervariables=True,
#     loadrecoveryvars=True,
#     loadsystematics=True,
#     loadnumuvariables=True, # True for 1mu1p/ False for 1e1p ####NOTE: For running full sample numupresel must always be off!!!!
#     numupresel=False,
#     # Load the nu_e set one more time with the LEE weights applied
#     load_lee=False,   
#     enable_cache=True,
#     # Since this is Open Data, we are allowed to unblind the data. By default, the data is blinded.
#     blinded=blinded,
#     load_numu_tki=True,
#     load_nue_tki=True
# )

# #For 1e1p
# rundata_nue, mc_weights_nue, data_pot_nue = load_runs(
#     RUN,
#     data="bnb",      
#     truth_filtered_sets=["nue", "drt"],
#     loadpi0variables=True,
#     loadshowervariables=True,
#     loadrecoveryvars=True,
#     loadsystematics=True,
#     loadnumuvariables=False, # True for 1mu1p/ False for 1e1p
#     numupresel=False,
#     load_lee=False,
#     enable_cache=True,
#     blinded=blinded,
#     load_numu_tki=True,
#     load_nue_tki=True
# )

