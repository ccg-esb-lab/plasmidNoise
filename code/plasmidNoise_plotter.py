"""
plasmidNoise_plotter.py

Plotting functions for plasmidNoise simulations.

These functions expect a SimulationResult object produced by plasmidNoise_model.py.
They do not run simulations and do not modify saved results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plasmidNoise_model import SimulationResult, transfer_end_table


PathLike = str | Path


def _maybe_save(fig, save_path: Optional[PathLike] = None, dpi: int = 300):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def _strain_color_map(result: SimulationResult) -> dict:
    if "color" in result.parameter_table.columns:
        return dict(zip(result.parameter_table["strain"].astype(str), result.parameter_table["color"].astype(str)))
    return {}


def plot_environment(
    environment: Sequence[float],
    *,
    ax=None,
    title: str = "Environment",
    ylabel: str = "Antibiotic",
    save_path: Optional[PathLike] = None,
):
    """Plot environmental antibiotic concentration across transfers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.5))
    else:
        fig = ax.figure
    env = np.asarray(environment, dtype=float)
    ax.plot(np.arange(len(env)), env, marker="o", linewidth=1)
    ax.set_xlabel("Transfer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _maybe_save(fig, save_path)


def plot_environment_strip(
    environment: Sequence[float],
    *,
    ax=None,
    vmax: Optional[float] = None,
    title: str = "Environment",
    save_path: Optional[PathLike] = None,
):
    """Plot environmental concentration as a heat strip."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.8))
    else:
        fig = ax.figure
    env = np.asarray(environment, dtype=float)
    vmax = float(vmax) if vmax is not None else float(np.nanmax(env) if len(env) else 1.0)
    norm = mcolors.Normalize(vmin=0, vmax=max(vmax, 1e-12))
    ax.imshow([env], cmap="gray_r", norm=norm, aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Transfer")
    ax.set_title(title)
    return _maybe_save(fig, save_path)


def plot_transfer_summary(
    result: SimulationResult,
    *,
    columns: Sequence[str] = ("B_total_end", "Bp_total_end", "B0_total_end"),
    log_y: bool = True,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Plot one or more endpoint summary variables across transfers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    df = result.transfer_summary
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column not found in transfer_summary: {col}")
        ax.plot(df["transfer"], df[col], marker="o", linewidth=1, label=col)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Transfer")
    ax.set_ylabel("Endpoint value")
    ax.set_title(result.run_id)
    ax.legend(frameon=False)
    return _maybe_save(fig, save_path)


def plot_total_density_timecourse(
    result: SimulationResult,
    *,
    transfers: Optional[Sequence[int]] = None,
    log_y: bool = True,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Plot total density within selected transfers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    if transfers is None:
        transfers = [0, len(result.transfers) - 1] if len(result.transfers) > 1 else [0]
    for idx in transfers:
        tr = result.transfers[int(idx)]
        n = len(result.members)
        y = tr["y"]
        total = y[2 : 2 + 2 * n, :].sum(axis=0)
        ax.plot(tr["t"], total, linewidth=1.5, label=f"transfer {idx}; A={tr['A_initial']:.3g}")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Time within transfer")
    ax.set_ylabel("Total density")
    ax.set_title(result.run_id)
    ax.legend(frameon=False)
    return _maybe_save(fig, save_path)


def plot_plasmid_fraction_summary(
    result: SimulationResult,
    *,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Plot total plasmid fraction at the end of each transfer."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = ax.figure
    df = result.transfer_summary
    ax.plot(df["transfer"], df["plasmid_fraction_end"], marker="o", linewidth=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Transfer")
    ax.set_ylabel("Plasmid fraction")
    ax.set_title(result.run_id)
    return _maybe_save(fig, save_path)


def plot_plasmid_fraction_by_strain(
    result: SimulationResult,
    *,
    strains: Optional[Sequence[str]] = None,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Plot endpoint plasmid fraction for each strain across transfers."""
    df = transfer_end_table(result)
    if strains is not None:
        df = df[df["strain"].isin([str(s) for s in strains])]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    colors = _strain_color_map(result)
    for strain, gdf in df.groupby("strain", sort=False):
        kwargs = {"color": colors[strain]} if strain in colors else {}
        ax.plot(gdf["transfer"], gdf["plasmid_fraction_end"], linewidth=1, label=strain, **kwargs)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Transfer")
    ax.set_ylabel("Plasmid fraction")
    ax.set_title(result.run_id)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _maybe_save(fig, save_path)


def plot_density_by_strain(
    result: SimulationResult,
    *,
    strains: Optional[Sequence[str]] = None,
    log_y: bool = True,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Plot endpoint total density per strain across transfers."""
    df = transfer_end_table(result)
    if strains is not None:
        df = df[df["strain"].isin([str(s) for s in strains])]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    colors = _strain_color_map(result)
    for strain, gdf in df.groupby("strain", sort=False):
        kwargs = {"color": colors[strain]} if strain in colors else {}
        ax.plot(gdf["transfer"], gdf["B_total_end"], linewidth=1, label=strain, **kwargs)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Transfer")
    ax.set_ylabel("Total density")
    ax.set_title(result.run_id)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _maybe_save(fig, save_path)


def plot_plasmid_fraction_with_environment(
    result: SimulationResult,
    *,
    strains: Optional[Sequence[str]] = None,
    environment_vmax: Optional[float] = None,
    save_path: Optional[PathLike] = None,
):
    """Two-panel plot: environment heat strip plus per-strain plasmid fraction."""
    fig, (ax_env, ax_pf) = plt.subplots(
        2,
        1,
        figsize=(8, 5),
        gridspec_kw={"height_ratios": [1, 5]},
        sharex=True,
    )
    plot_environment_strip(result.environment, ax=ax_env, vmax=environment_vmax, title="")
    ax_env.set_xlabel("")
    plot_plasmid_fraction_by_strain(result, strains=strains, ax=ax_pf)
    ax_pf.set_title(result.run_id)
    return _maybe_save(fig, save_path)


def plot_final_composition(
    result: SimulationResult,
    *,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Plot final Bp and B0 density per strain."""
    df = result.final_state.copy()
    x = np.arange(len(df))
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.35), 4))
    else:
        fig = ax.figure
    ax.bar(x, df["B0"], label="B0 plasmid-free")
    ax.bar(x, df["Bp"], bottom=df["B0"], label="Bp plasmid-bearing")
    ax.set_xticks(x)
    ax.set_xticklabels(df["strain"], rotation=90)
    ax.set_ylabel("Final density")
    ax.set_title(result.run_id)
    ax.legend(frameon=False)
    return _maybe_save(fig, save_path)


def plot_many_runs_summary(
    summaries: pd.DataFrame,
    *,
    x: str,
    y: str = "final_plasmid_fraction",
    hue: Optional[str] = None,
    ax=None,
    save_path: Optional[PathLike] = None,
):
    """Scatter plot for comparing many saved simulations summarized in one dataframe."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    if hue is None:
        ax.scatter(summaries[x], summaries[y])
    else:
        for label, gdf in summaries.groupby(hue):
            ax.scatter(gdf[x], gdf[y], label=str(label))
        ax.legend(frameon=False)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return _maybe_save(fig, save_path)
