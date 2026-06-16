"""
plasmidNoise_model.py

Core model and simulator functions for plasmid dynamics in microbial communities.

State vector convention
-----------------------
y = [R, A, Bp_1, ..., Bp_N, B0_1, ..., B0_N]

where
    R    = limiting resource concentration
    A    = extracellular antibiotic concentration
    Bp_i = plasmid-bearing density of strain i
    B0_i = plasmid-free density of strain i

Recommended workflow
--------------------
1. Load parameters from CSV.
2. Create or load an environmental antibiotic trajectory.
3. Run simulation.
4. Save simulation to runs/.
5. Reload simulation later for plotting or analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import gzip
import json
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


ArrayLike = Union[np.ndarray, Sequence[float], List[float]]
PathLike = Union[str, Path]
MemberLike = Union[str, int]


# -----------------------------------------------------------------------------
# Configuration and result containers
# -----------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Settings for serial-transfer simulations."""

    season_duration: float = 24.0
    points_per_season: int = 241
    R0: float = 1.0
    B0_total: float = 1e6
    initial_plasmid_fraction: float = 1.0
    dilution: float = 0.1
    extinction_threshold: float = 1.0
    solver_method: str = "LSODA"
    rtol: float = 1e-7
    atol: float = 1e-9
    max_step: Optional[float] = None
    clip_negative: bool = True
    uptake_mode: str = "monod"  # "monod" or "linear_affinity"


@dataclass
class SimulationResult:
    """Container returned by simulate_serial_transfers."""

    run_id: str
    members: List[str]
    config: Dict[str, Any]
    parameter_table: pd.DataFrame
    environment: np.ndarray
    transfers: List[Dict[str, Any]]
    transfer_summary: pd.DataFrame
    final_state: pd.DataFrame
    metadata: Dict[str, Any]


# -----------------------------------------------------------------------------
# Project paths and small utilities
# -----------------------------------------------------------------------------

def setup_project_paths(
    project_root: PathLike,
    *,
    add_code_to_syspath: bool = True,
    make_dirs: bool = True,
) -> Dict[str, Path]:
    """
    Create and return standard project paths.

    Expected structure:
        project_root/
            code/
            data/
            env/
            runs/
            figures/
    """
    import sys

    root = Path(project_root).expanduser().resolve()
    paths = {
        "root": root,
        "code": root / "code",
        "data": root / "data",
        "env": root / "env",
        "runs": root / "runs",
        "figures": root / "figures",
    }
    if make_dirs:
        for key in ["code", "data", "env", "runs", "figures"]:
            paths[key].mkdir(parents=True, exist_ok=True)
    if add_code_to_syspath and str(paths["code"]) not in sys.path:
        sys.path.insert(0, str(paths["code"]))
    return paths


def get_sequences_from_files(directory: PathLike, length: Optional[int] = None) -> List[List[int]]:
    """
    Extract integer sequences from files named like sim_Es_1_2_3.pkl or sim_Es_1_2_3.pkl.gz.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    sequences: List[List[int]] = []
    pattern = re.compile(r"^sim_Es_([0-9_]+)\.pkl(?:\.gz)?$")
    for filename in os.listdir(directory):
        m = pattern.match(filename)
        if not m:
            continue
        seq = [int(x) for x in m.group(1).split("_") if x != ""]
        if length is None or len(seq) == length:
            sequences.append(seq)
    return sequences


# -----------------------------------------------------------------------------
# Parameter loading and validation
# -----------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "strain": ["strain", "strain_id", "id", "isolate", "host", "member"],
    "display_name": ["name", "display_name", "label"],
    "species": ["specie", "species"],
    "color": ["color", "colour"],
    "status": ["status", "state", "plasmid_status", "subpopulation", "type"],
    "r": ["r", "rho", "growth_efficiency", "yield", "conversion", "resource_conversion"],
    "Vmax": ["Vmax", "vmax", "V_max", "v_max"],
    "Km": ["Km", "km", "K_m", "k_m"],
    "affinity": ["affinity", "VKm", "V/Km", "Vmax_over_Km", "Vmax_Km", "V_Km", "vmax_over_km", "specific_affinity"],
    "alpha": ["alpha", "a", "antibiotic_degradation", "drug_degradation"],
    "k": ["k", "kill", "kill_rate", "antibiotic_killing"],
    "k_star": ["k_star", "kstar", "k*", "kappa", "resistance", "resistance_parameter"],
    "lambda_loss": ["lambda_loss", "lambda", "l", "loss", "seg_rate", "segregation", "segregational_loss"],
    "gamma": ["gamma", "g", "conj_rate", "conjugation", "conjugation_rate", "permissiveness"],
    "Bp0": ["Bp0", "B_p0", "initial_Bp", "initial_plasmid", "initial_TC"],
    "B00": ["B00", "B_00", "B0_0", "initial_B0", "initial_free", "initial_WT"],
}


def _find_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    aliases = _COLUMN_ALIASES.get(canonical, [canonical])
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def _coalesce_wide_column(df: pd.DataFrame, base: str, suffixes: Sequence[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    lower_to_original = {str(c).lower(): c for c in df.columns}
    base_aliases = _COLUMN_ALIASES.get(base, [base])
    for suffix in suffixes:
        candidates: List[str] = []
        for b in base_aliases:
            candidates.extend([
                f"{b}_{suffix}", f"{b}{suffix}", f"{suffix}_{b}",
                f"{b}_{'free' if suffix == '0' else 'plasmid'}",
                f"{b}_{'WT' if suffix == '0' else 'TC'}",
                f"{b}_{'B0' if suffix == '0' else 'Bp'}",
            ])
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand
                break
            if cand.lower() in lower_to_original:
                found = lower_to_original[cand.lower()]
                break
        out[suffix] = found
    return out


def _status_to_suffix(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"p", "bp", "plasmid", "plasmid-bearing", "plasmid_bearing", "tc", "transconjugant", "resistant"}:
        return "p"
    if s in {"0", "b0", "free", "plasmid-free", "plasmid_free", "wt", "wildtype", "wild-type", "susceptible"}:
        return "0"
    raise ValueError(f"Cannot interpret plasmid status value: {value}")


def _maybe_log10_to_linear(values: pd.Series) -> pd.Series:
    """Convert log10 rates to linear rates when all finite values are negative."""
    x = pd.to_numeric(values, errors="coerce")
    finite = x.dropna()
    if len(finite) > 0 and (finite < 0).all():
        return 10.0 ** x
    return x


def load_parameter_table(
    csv_path: PathLike,
    *,
    mic_max: Optional[float] = None,
    default_Km: float = 1.0,
    default_alpha_p: float = 1e-12,
    default_alpha_0: float = 1e-10,
    default_lambda_loss: float = 0.0,
    default_gamma: float = 0.0,
    k_star_is_resistance: bool = True,
    conjugation_log10_auto: bool = True,
) -> pd.DataFrame:
    """
    Load strain parameters from CSV and return a standardized wide table.

    Supported formats
    -----------------
    1. Wide format, one row per strain:
       strain,r_p,r_0,Vmax_p,Vmax_0,Km_p,Km_0,alpha_p,alpha_0,k_p,k_0,lambda_loss,gamma

    2. Long format, one row per strain/status:
       strain,type,rho,VKm,kappa,seg_rate,conj_rate
       where type can be TC/plasmid-bearing or WT/plasmid-free.

    For the pOXA-48 file, the loader uses:
        rho      -> r
        VKm      -> affinity, with Km=1 and Vmax=VKm
        kappa    -> resistance parameter; kill rate k = 1 / (kappa * mic_max)
        seg_rate -> lambda_loss
        conj_rate -> gamma; negative values are interpreted as log10(gamma)

    If mic_max is not supplied and MIC/kappa columns are available, mic_max is inferred as median(MIC/kappa).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError(f"Parameter CSV is empty: {csv_path}")

    strain_col = _find_column(raw, "strain")
    if strain_col is None:
        raise ValueError("Parameter table must contain a strain identifier column.")

    raw = raw.copy()
    gamma_col = _find_column(raw, "gamma")
    if gamma_col is not None and conjugation_log10_auto:
        raw[gamma_col] = _maybe_log10_to_linear(raw[gamma_col])

    if mic_max is None:
        kstar_col = _find_column(raw, "k_star")
        mic_col = "MIC" if "MIC" in raw.columns else None
        if kstar_col is not None and mic_col is not None:
            ratio = pd.to_numeric(raw[mic_col], errors="coerce") / pd.to_numeric(raw[kstar_col], errors="coerce")
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) > 0:
                mic_max = float(np.median(ratio))

    status_col = _find_column(raw, "status")
    if status_col is not None:
        params = _standardize_long_parameter_table(
            raw,
            strain_col,
            status_col,
            mic_max=mic_max,
            default_Km=default_Km,
            default_alpha_p=default_alpha_p,
            default_alpha_0=default_alpha_0,
            default_lambda_loss=default_lambda_loss,
            default_gamma=default_gamma,
            k_star_is_resistance=k_star_is_resistance,
        )
    else:
        params = _standardize_wide_parameter_table(
            raw,
            strain_col,
            mic_max=mic_max,
            default_Km=default_Km,
            default_alpha_p=default_alpha_p,
            default_alpha_0=default_alpha_0,
            default_lambda_loss=default_lambda_loss,
            default_gamma=default_gamma,
            k_star_is_resistance=k_star_is_resistance,
        )

    params["strain"] = params["strain"].astype(str)
    params = params.drop_duplicates(subset=["strain"], keep="first").reset_index(drop=True)
    _validate_parameter_table(params)
    return params


def _kill_from_kstar(kstar: pd.Series, mic_max: Optional[float], *, k_star_is_resistance: bool) -> pd.Series:
    kstar = pd.to_numeric(kstar, errors="coerce")
    if not k_star_is_resistance:
        return kstar
    if mic_max is None:
        raise ValueError("Found k_star/kappa columns but mic_max could not be inferred or was not provided.")
    return 1.0 / (kstar * float(mic_max))


def _standardize_wide_parameter_table(
    raw: pd.DataFrame,
    strain_col: str,
    *,
    mic_max: Optional[float],
    default_Km: float,
    default_alpha_p: float,
    default_alpha_0: float,
    default_lambda_loss: float,
    default_gamma: float,
    k_star_is_resistance: bool,
) -> pd.DataFrame:
    out = pd.DataFrame({"strain": raw[strain_col].astype(str)})

    for meta in ["display_name", "species", "color"]:
        col = _find_column(raw, meta)
        if col is not None:
            out[meta] = raw[col]

    for base in ["r", "Vmax", "Km", "affinity", "alpha", "k", "k_star"]:
        cols = _coalesce_wide_column(raw, base, ["p", "0"])
        for suffix, col in cols.items():
            if col is not None:
                out[f"{base}_{suffix}"] = pd.to_numeric(raw[col], errors="coerce")

    for base, default in [("lambda_loss", default_lambda_loss), ("gamma", default_gamma), ("Bp0", np.nan), ("B00", np.nan)]:
        col = _find_column(raw, base)
        out[base] = pd.to_numeric(raw[col], errors="coerce") if col is not None else default

    _fill_defaults_and_conversions(out, mic_max, default_Km, default_alpha_p, default_alpha_0, k_star_is_resistance)
    return out


def _standardize_long_parameter_table(
    raw: pd.DataFrame,
    strain_col: str,
    status_col: str,
    *,
    mic_max: Optional[float],
    default_Km: float,
    default_alpha_p: float,
    default_alpha_0: float,
    default_lambda_loss: float,
    default_gamma: float,
    k_star_is_resistance: bool,
) -> pd.DataFrame:
    df = raw.copy()
    df["_strain"] = df[strain_col].astype(str)
    df["_suffix"] = df[status_col].map(_status_to_suffix)

    value_cols: Dict[str, str] = {}
    for base in ["r", "Vmax", "Km", "affinity", "alpha", "k", "k_star"]:
        col = _find_column(df, base)
        if col is not None:
            value_cols[base] = col

    meta_cols = {meta: _find_column(df, meta) for meta in ["display_name", "species", "color"]}
    lambda_col = _find_column(df, "lambda_loss")
    gamma_col = _find_column(df, "gamma")
    bp0_col = _find_column(df, "Bp0")
    b00_col = _find_column(df, "B00")

    rows: List[Dict[str, Any]] = []
    for strain, gdf in df.groupby("_strain", sort=False):
        row: Dict[str, Any] = {"strain": strain}
        for meta, col in meta_cols.items():
            if col is not None:
                # Prefer the plasmid-bearing row for display metadata when duplicated.
                preferred = gdf[gdf["_suffix"] == "p"]
                row[meta] = (preferred if not preferred.empty else gdf).iloc[0][col]

        for suffix in ["p", "0"]:
            sdf = gdf[gdf["_suffix"] == suffix]
            if sdf.empty:
                raise ValueError(f"Missing status {suffix} for strain {strain}")
            srow = sdf.iloc[0]
            for base, col in value_cols.items():
                row[f"{base}_{suffix}"] = pd.to_numeric(srow[col], errors="coerce")

        # Segregational loss is a plasmid-bearing property. Prefer the p/TC row.
        if lambda_col is not None:
            pvals = gdf.loc[gdf["_suffix"] == "p", lambda_col]
            vals = pvals if pvals.notna().any() else gdf[lambda_col]
            row["lambda_loss"] = pd.to_numeric(vals.dropna().iloc[0], errors="coerce") if vals.notna().any() else default_lambda_loss
        else:
            row["lambda_loss"] = default_lambda_loss

        # Conjugation/permissiveness is a recipient property. Prefer the 0/WT row.
        if gamma_col is not None:
            zvals = gdf.loc[gdf["_suffix"] == "0", gamma_col]
            vals = zvals if zvals.notna().any() else gdf[gamma_col]
            row["gamma"] = pd.to_numeric(vals.dropna().iloc[0], errors="coerce") if vals.notna().any() else default_gamma
        else:
            row["gamma"] = default_gamma

        if bp0_col is not None:
            pvals = gdf.loc[gdf["_suffix"] == "p", bp0_col]
            row["Bp0"] = pd.to_numeric(pvals.dropna().iloc[0], errors="coerce") if pvals.notna().any() else np.nan
        if b00_col is not None:
            zvals = gdf.loc[gdf["_suffix"] == "0", b00_col]
            row["B00"] = pd.to_numeric(zvals.dropna().iloc[0], errors="coerce") if zvals.notna().any() else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    _fill_defaults_and_conversions(out, mic_max, default_Km, default_alpha_p, default_alpha_0, k_star_is_resistance)
    return out


def _fill_defaults_and_conversions(
    out: pd.DataFrame,
    mic_max: Optional[float],
    default_Km: float,
    default_alpha_p: float,
    default_alpha_0: float,
    k_star_is_resistance: bool,
) -> None:
    for suffix in ["p", "0"]:
        if f"Km_{suffix}" not in out:
            out[f"Km_{suffix}"] = default_Km
        if f"Vmax_{suffix}" not in out:
            if f"affinity_{suffix}" in out:
                out[f"Vmax_{suffix}"] = out[f"affinity_{suffix}"] * out[f"Km_{suffix}"]
            else:
                out[f"Vmax_{suffix}"] = 1.0
        if f"alpha_{suffix}" not in out:
            out[f"alpha_{suffix}"] = default_alpha_p if suffix == "p" else default_alpha_0
        if f"k_{suffix}" not in out:
            if f"k_star_{suffix}" in out:
                out[f"k_{suffix}"] = _kill_from_kstar(out[f"k_star_{suffix}"], mic_max, k_star_is_resistance=k_star_is_resistance)
            else:
                raise ValueError(f"Missing killing parameter k_{suffix} or k_star/kappa_{suffix}.")


def _validate_parameter_table(params: pd.DataFrame) -> None:
    required = [
        "strain", "r_p", "r_0", "Vmax_p", "Vmax_0", "Km_p", "Km_0",
        "alpha_p", "alpha_0", "k_p", "k_0", "lambda_loss", "gamma",
    ]
    missing = [c for c in required if c not in params.columns]
    if missing:
        raise ValueError(f"Parameter table is missing required standardized columns: {missing}")

    numeric_cols = [c for c in required if c != "strain"]
    bad = params[numeric_cols].isna().any()
    bad_cols = list(bad[bad].index)
    if bad_cols:
        raise ValueError(f"Parameter table contains NaN values in required columns: {bad_cols}")

    if (params[["Km_p", "Km_0"]] <= 0).any().any():
        raise ValueError("Km values must be positive.")
    nonneg = ["r_p", "r_0", "Vmax_p", "Vmax_0", "alpha_p", "alpha_0", "k_p", "k_0", "lambda_loss", "gamma"]
    if (params[nonneg] < 0).any().any():
        raise ValueError("Growth, degradation, killing, loss, and conjugation parameters must be nonnegative.")


def import_model_params(csv_path: PathLike, expe_params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> pd.DataFrame:
    """Backward-compatible wrapper around load_parameter_table."""
    if expe_params:
        if "MICmax" in expe_params and "mic_max" not in kwargs:
            kwargs["mic_max"] = expe_params["MICmax"]
        if "mic_max" in expe_params and "mic_max" not in kwargs:
            kwargs["mic_max"] = expe_params["mic_max"]
        if "R0" in expe_params and "default_Km" not in kwargs:
            # Do not force this by default; kept only for compatibility if users pass it explicitly elsewhere.
            pass
    return load_parameter_table(csv_path, **kwargs)


def display_model_params_stats(params: pd.DataFrame, strains_subset: Optional[Sequence[MemberLike]] = None) -> pd.DataFrame:
    """Print and return compact parameter statistics."""
    df = select_members(params, strains_subset) if strains_subset is not None else params
    cols = [c for c in ["r_p", "r_0", "Vmax_p", "Vmax_0", "k_p", "k_0", "lambda_loss", "gamma"] if c in df]
    stats = df[cols].describe().T
    print(f"Number of strains: {len(df)}")
    print(stats)
    return stats


def select_members(params: pd.DataFrame, members: Sequence[MemberLike]) -> pd.DataFrame:
    """Select strains by name or integer row position."""
    if members is None or len(members) == 0:
        raise ValueError("members must contain at least one strain name or integer index.")
    rows = []
    for m in members:
        if isinstance(m, (int, np.integer)):
            if int(m) < 0 or int(m) >= len(params):
                raise IndexError(f"Member index out of range: {m}")
            rows.append(params.iloc[int(m)])
        else:
            hit = params[params["strain"].astype(str) == str(m)]
            if hit.empty and "display_name" in params.columns:
                hit = params[params["display_name"].astype(str) == str(m)]
            if hit.empty:
                raise KeyError(f"Strain not found in parameter table: {m}")
            rows.append(hit.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Environment loading / creation
# -----------------------------------------------------------------------------

def load_environment_txt(path: PathLike, *, scale: float = 1.0) -> np.ndarray:
    """Load an antibiotic trajectory from a txt/csv-like file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Environment file not found: {path}")
    try:
        arr = np.loadtxt(path, delimiter=None)
    except ValueError:
        arr = np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"Environment file is empty: {path}")
    if np.any(arr < 0):
        raise ValueError("Environment concentrations must be nonnegative.")
    return scale * arr


def load_environments(folder: PathLike, prefix: str, indices: Iterable[int], *, scale: float = 1.0) -> Dict[int, np.ndarray]:
    """Load multiple environment files from a folder."""
    folder = Path(folder)
    out: Dict[int, np.ndarray] = {}
    for i in indices:
        candidates = [
            folder / f"{prefix}_{i}.txt", folder / f"{prefix}{i}.txt",
            folder / f"{prefix}_{i}.csv", folder / f"{prefix}{i}.csv",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Could not find environment {i}. Tried: {candidates}")
        out[int(i)] = load_environment_txt(found, scale=scale)
    return out


def make_constant_environment(n_transfers: int, concentration: float) -> np.ndarray:
    if n_transfers <= 0:
        raise ValueError("n_transfers must be positive.")
    if concentration < 0:
        raise ValueError("concentration must be nonnegative.")
    return np.full(int(n_transfers), float(concentration))


def make_periodic_environment(n_transfers: int, low: float, high: float, period: int, duty_cycle: float = 0.5) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be positive.")
    if not (0 <= duty_cycle <= 1):
        raise ValueError("duty_cycle must be between 0 and 1.")
    env = np.empty(int(n_transfers), dtype=float)
    on_steps = int(round(period * duty_cycle))
    for t in range(int(n_transfers)):
        env[t] = high if (t % period) < on_steps else low
    return env


def make_stochastic_environment(
    n_transfers: int,
    mean: float,
    sd: float,
    *,
    seed: Optional[int] = None,
    distribution: str = "normal",
    clip_min: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        env = rng.normal(loc=mean, scale=sd, size=int(n_transfers))
    elif distribution == "lognormal":
        if mean <= 0:
            raise ValueError("mean must be positive for lognormal environments.")
        env = rng.lognormal(mean=np.log(mean), sigma=sd, size=int(n_transfers))
    elif distribution == "uniform":
        env = rng.uniform(low=max(0.0, mean - sd), high=mean + sd, size=int(n_transfers))
    else:
        raise ValueError("distribution must be 'normal', 'lognormal', or 'uniform'.")
    return np.clip(env, clip_min, None)


# -----------------------------------------------------------------------------
# Model equations
# -----------------------------------------------------------------------------

def uptake(R: float, Vmax: np.ndarray, Km: np.ndarray, *, mode: str = "monod") -> np.ndarray:
    R = max(float(R), 0.0)
    if mode == "monod":
        return Vmax * R / (R + Km)
    if mode == "linear_affinity":
        return (Vmax / Km) * R
    raise ValueError("uptake mode must be 'monod' or 'linear_affinity'.")


def growth(R: float, r: np.ndarray, Vmax: np.ndarray, Km: np.ndarray, *, mode: str = "monod") -> np.ndarray:
    return r * uptake(R, Vmax, Km, mode=mode)


def plasmid_noise_rhs(t: float, y: np.ndarray, strain_params: pd.DataFrame, *, uptake_mode: str = "monod") -> np.ndarray:
    """Right-hand side for the multistrain plasmid dynamics model."""
    n = len(strain_params)
    R = max(y[0], 0.0)
    A = max(y[1], 0.0)
    Bp = np.clip(y[2 : 2 + n], 0.0, None)
    B0 = np.clip(y[2 + n : 2 + 2 * n], 0.0, None)

    r_p = strain_params["r_p"].to_numpy(float)
    r_0 = strain_params["r_0"].to_numpy(float)
    Vmax_p = strain_params["Vmax_p"].to_numpy(float)
    Vmax_0 = strain_params["Vmax_0"].to_numpy(float)
    Km_p = strain_params["Km_p"].to_numpy(float)
    Km_0 = strain_params["Km_0"].to_numpy(float)
    alpha_p = strain_params["alpha_p"].to_numpy(float)
    alpha_0 = strain_params["alpha_0"].to_numpy(float)
    k_p = strain_params["k_p"].to_numpy(float)
    k_0 = strain_params["k_0"].to_numpy(float)
    lam = strain_params["lambda_loss"].to_numpy(float)
    gamma = strain_params["gamma"].to_numpy(float)

    U_p = uptake(R, Vmax_p, Km_p, mode=uptake_mode)
    U_0 = uptake(R, Vmax_0, Km_0, mode=uptake_mode)
    G_p = r_p * U_p
    G_0 = r_0 * U_0
    donor_pool = np.sum(Bp)

    dR = -np.sum(U_p * Bp + U_0 * B0)
    dA = -A * np.sum(alpha_p * Bp + alpha_0 * B0)
    dB0 = (G_0 - k_0 * A) * B0 + lam * G_p * Bp - gamma * B0 * donor_pool
    dBp = (1.0 - lam) * G_p * Bp - k_p * A * Bp + gamma * B0 * donor_pool

    return np.concatenate(([dR, dA], dBp, dB0))


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

def build_initial_state(
    strain_params: pd.DataFrame,
    *,
    R0: float,
    A0: float,
    B0_total: float,
    initial_plasmid_fraction: float,
) -> np.ndarray:
    """Build initial state vector for selected community members."""
    n = len(strain_params)
    if n == 0:
        raise ValueError("strain_params must contain at least one strain.")
    if not (0 <= initial_plasmid_fraction <= 1):
        raise ValueError("initial_plasmid_fraction must be between 0 and 1.")

    if {"Bp0", "B00"}.issubset(strain_params.columns) and strain_params[["Bp0", "B00"]].notna().all().all():
        Bp = strain_params["Bp0"].to_numpy(float)
        B0 = strain_params["B00"].to_numpy(float)
    else:
        per_strain = float(B0_total) / n
        Bp = np.full(n, per_strain * initial_plasmid_fraction, dtype=float)
        B0 = np.full(n, per_strain * (1.0 - initial_plasmid_fraction), dtype=float)

    return np.concatenate(([float(R0), float(A0)], Bp, B0))


def simulate_one_transfer(y0: np.ndarray, strain_params: pd.DataFrame, config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one growth season / transfer."""
    t_eval = np.linspace(0.0, config.season_duration, config.points_per_season)
    kwargs: Dict[str, Any] = dict(
        fun=lambda t, y: plasmid_noise_rhs(t, y, strain_params, uptake_mode=config.uptake_mode),
        t_span=(0.0, config.season_duration),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method=config.solver_method,
        rtol=config.rtol,
        atol=config.atol,
    )
    if config.max_step is not None:
        kwargs["max_step"] = config.max_step
    sol = solve_ivp(**kwargs)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    y = np.clip(sol.y, 0.0, None) if config.clip_negative else sol.y
    return sol.t, y


def _prepare_environment(environment: Union[ArrayLike, Callable[[int], float]], n_transfers: Optional[int]) -> np.ndarray:
    if callable(environment):
        if n_transfers is None:
            raise ValueError("n_transfers is required when environment is a function.")
        env = np.array([float(environment(i)) for i in range(int(n_transfers))], dtype=float)
    else:
        env = np.asarray(environment, dtype=float).ravel()
        if n_transfers is not None:
            env = env[: int(n_transfers)]
    if env.size == 0:
        raise ValueError("environment must contain at least one concentration.")
    if np.any(env < 0):
        raise ValueError("environment concentrations must be nonnegative.")
    return env


def simulate_serial_transfers(
    parameter_table: pd.DataFrame,
    members: Sequence[MemberLike],
    environment: Union[ArrayLike, Callable[[int], float]],
    *,
    config: Optional[SimulationConfig] = None,
    n_transfers: Optional[int] = None,
    run_id: str = "simulation",
    metadata: Optional[Mapping[str, Any]] = None,
) -> SimulationResult:
    """Simulate plasmid dynamics across serial transfers."""
    config = config or SimulationConfig()
    env = _prepare_environment(environment, n_transfers)
    strain_params = select_members(parameter_table, members)
    member_names = strain_params["strain"].astype(str).tolist()

    y0 = build_initial_state(
        strain_params,
        R0=config.R0,
        A0=env[0],
        B0_total=config.B0_total,
        initial_plasmid_fraction=config.initial_plasmid_fraction,
    )

    transfers: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for transfer_idx, A_initial in enumerate(env):
        y0[0] = config.R0
        y0[1] = float(A_initial)

        t, y = simulate_one_transfer(y0, strain_params, config)
        n = len(strain_params)
        Bp_end = np.clip(y[2 : 2 + n, -1], 0.0, None)
        B0_end = np.clip(y[2 + n : 2 + 2 * n, -1], 0.0, None)
        total_end = Bp_end + B0_end

        Bp_next = Bp_end * config.dilution
        B0_next = B0_end * config.dilution
        Bp_next[Bp_next < config.extinction_threshold] = 0.0
        B0_next[B0_next < config.extinction_threshold] = 0.0

        transfers.append({
            "transfer": int(transfer_idx),
            "A_initial": float(A_initial),
            "t": t,
            "y": y,
            "members": member_names,
        })
        summaries.append({
            "transfer": int(transfer_idx),
            "A_initial": float(A_initial),
            "R_end": float(y[0, -1]),
            "A_end": float(y[1, -1]),
            "Bp_total_end": float(Bp_end.sum()),
            "B0_total_end": float(B0_end.sum()),
            "B_total_end": float(total_end.sum()),
            "plasmid_fraction_end": float(Bp_end.sum() / total_end.sum()) if total_end.sum() > 0 else np.nan,
            "richness_end": int(np.sum(total_end > config.extinction_threshold)),
            "shannon_strain_end": shannon_index(total_end),
        })

        y0 = np.concatenate(([config.R0, 0.0], Bp_next, B0_next))

    transfer_summary = pd.DataFrame(summaries)
    final_state = final_state_table(transfers[-1], strain_params)

    return SimulationResult(
        run_id=run_id,
        members=member_names,
        config=asdict(config),
        parameter_table=strain_params,
        environment=env,
        transfers=transfers,
        transfer_summary=transfer_summary,
        final_state=final_state,
        metadata=dict(metadata or {}),
    )


# -----------------------------------------------------------------------------
# Analysis helpers
# -----------------------------------------------------------------------------

def transfer_end_table(result: SimulationResult) -> pd.DataFrame:
    """Return one row per strain per transfer with endpoint densities."""
    rows: List[Dict[str, Any]] = []
    for tr in result.transfers:
        n = len(tr["members"])
        y = tr["y"]
        Bp = y[2 : 2 + n, -1]
        B0 = y[2 + n : 2 + 2 * n, -1]
        for i, strain in enumerate(tr["members"]):
            total = Bp[i] + B0[i]
            rows.append({
                "transfer": tr["transfer"],
                "strain": strain,
                "A_initial": tr["A_initial"],
                "Bp_end": float(Bp[i]),
                "B0_end": float(B0[i]),
                "B_total_end": float(total),
                "plasmid_fraction_end": float(Bp[i] / total) if total > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def final_state_table(transfer: Dict[str, Any], strain_params: pd.DataFrame) -> pd.DataFrame:
    """Return final state for each strain from one transfer dictionary."""
    n = len(strain_params)
    y = transfer["y"]
    Bp = y[2 : 2 + n, -1]
    B0 = y[2 + n : 2 + 2 * n, -1]
    total = Bp + B0
    df = pd.DataFrame({
        "strain": strain_params["strain"].astype(str).tolist(),
        "Bp": Bp,
        "B0": B0,
        "B_total": total,
        "plasmid_fraction": np.divide(Bp, total, out=np.full_like(Bp, np.nan, dtype=float), where=total > 0),
    })
    for meta in ["display_name", "species", "color"]:
        if meta in strain_params.columns:
            df[meta] = strain_params[meta].to_numpy()
    return df


def shannon_index(abundances: ArrayLike) -> float:
    x = np.asarray(abundances, dtype=float)
    x = np.clip(x, 0.0, None)
    total = x.sum()
    if total <= 0:
        return np.nan
    p = x[x > 0] / total
    return float(-np.sum(p * np.log(p)))


def auc_over_transfers(result: SimulationResult, column: str) -> float:
    if column not in result.transfer_summary:
        raise KeyError(f"Column not found in transfer_summary: {column}")
    x = result.transfer_summary["transfer"].to_numpy(float)
    y = result.transfer_summary[column].to_numpy(float)
    return float(np.trapezoid(y, x))


def summarize_simulation(result: SimulationResult) -> Dict[str, Any]:
    ts = result.transfer_summary
    final = ts.iloc[-1]
    return {
        "run_id": result.run_id,
        "n_members": len(result.members),
        "n_transfers": len(result.environment),
        "final_B_total": float(final["B_total_end"]),
        "final_Bp_total": float(final["Bp_total_end"]),
        "final_B0_total": float(final["B0_total_end"]),
        "final_plasmid_fraction": float(final["plasmid_fraction_end"]),
        "final_richness": int(final["richness_end"]),
        "final_shannon": float(final["shannon_strain_end"]),
        "mean_plasmid_fraction": float(ts["plasmid_fraction_end"].mean()),
        "auc_plasmid_fraction": auc_over_transfers(result, "plasmid_fraction_end"),
    }


# -----------------------------------------------------------------------------
# Save / load
# -----------------------------------------------------------------------------

def save_simulation(result: SimulationResult, output_path: PathLike, *, save_csv_summaries: bool = True) -> Path:
    """Save a simulation result as pickle.gz and optionally export CSV summaries."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not (str(output_path).endswith(".pkl.gz") or str(output_path).endswith(".pkl") or str(output_path).endswith(".pickle")):
        output_path = output_path.with_suffix(".pkl.gz")

    opener = gzip.open if str(output_path).endswith(".gz") else open
    with opener(output_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    if save_csv_summaries:
        stem = output_path.name.replace(".pkl.gz", "").replace(".pickle", "").replace(".pkl", "")
        result.transfer_summary.to_csv(output_path.parent / f"{stem}_transfer_summary.csv", index=False)
        result.final_state.to_csv(output_path.parent / f"{stem}_final_state.csv", index=False)
        transfer_end_table(result).to_csv(output_path.parent / f"{stem}_strain_transfer_endpoints.csv", index=False)
        with open(output_path.parent / f"{stem}_metadata.json", "w") as f:
            json.dump({
                "run_id": result.run_id,
                "members": result.members,
                "config": result.config,
                "metadata": result.metadata,
                "summary": summarize_simulation(result),
            }, f, indent=2)
    return output_path


def load_simulation(path: PathLike) -> SimulationResult:
    """Load a simulation result saved by save_simulation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Simulation file not found: {path}")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        result = pickle.load(f)
    if not isinstance(result, SimulationResult):
        warnings.warn("Loaded object is not a SimulationResult; returning it anyway.")
    return result
