import os, hashlib, inspect, pandas as pd
import data_loading as dl
from data_loading import cache_dataframe, generate_hash
from numu_tki.cc0pi_analyzer import apply_ccnp0pi_stv

def _hash_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1<<20), b""):
            h.update(ch)
    return h.hexdigest()

def _df_fp(df: pd.DataFrame) -> str:
    cols = [c for c in ("run","sub","evt") if c in df.columns]
    base = pd.util.hash_pandas_object(df[cols], index=False) if cols else pd.util.hash_pandas_object(df.index, index=True)
    return hashlib.md5(base.values.tobytes()).hexdigest()

def _code_fp():
    return hashlib.md5(inspect.getsource(apply_ccnp0pi_stv).encode()).hexdigest()

@cache_dataframe
def _apply_cc0pi_cached(df_in: pd.DataFrame, model_path: str, _identity: str=None) -> pd.DataFrame:
    out = apply_ccnp0pi_stv(df_in.copy(), model_path=model_path)
    print("Caching the CC0pi-BDT....")
    # de-dup columns for HDF
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out
#def _apply_cc0pi_cached(df_in: pd.DataFrame, model_path: str, _identity: str=None) -> pd.DataFrame:
#    return apply_ccnp0pi_stv(df_in.copy(), model_path=model_path)
'''
def load_runs_with_cc0pi_bdt(run_numbers, *, model_path, enable_cache=False, overwrite=False, **kwargs):
    rundata, weights, data_pot = dl.load_runs(run_numbers, enable_cache=enable_cache, **kwargs)
    mc = rundata.get("mc")
    if mc is not None:
        ident = generate_hash(df_fp=_df_fp(mc), model=os.path.abspath(model_path),
                              model_md5=_hash_file(model_path), code_md5=_code_fp())
        rundata["mc"] = _apply_cc0pi_cached(mc, model_path, _identity=ident,
                                            enable_cache=enable_cache, overwrite=overwrite)
    return rundata, weights, data_pot
'''
def load_runs_with_cc0pi_bdt(run_numbers, *, model_path, enable_cache=False, overwrite=False, **kwargs):
    # hard-disable the legacy in-pipeline BDT
    kwargs["use_bdt"] = False

    rundata, weights, data_pot = dl.load_runs(run_numbers, enable_cache=enable_cache, **kwargs)
    mc = rundata.get("mc")
    print("Running the CC0pi-BDT....")
    if mc is not None:
        ident = generate_hash(df_fp=_df_fp(mc), model=os.path.abspath(model_path),
                              model_md5=_hash_file(model_path), code_md5=_code_fp())
        rundata["mc"] = _apply_cc0pi_cached(mc, model_path, _identity=ident,
                                            enable_cache=enable_cache, overwrite=overwrite)
    return rundata, weights, data_pot

    
