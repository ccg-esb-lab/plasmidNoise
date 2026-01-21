import csv
import numpy as np
import os
import operator
import pandas as pd
import pickle
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tabulate import tabulate
import colorcet as cc
def dB_conj(B_WT, B_TC, conj_exponent):
    """
    Conjugation gain term for a WT recipient from plasmid-bearing donors.

    Notes
    -----
    `conj_exponent` is interpreted as log10(rate). If NaN, rate is set to 0
    (used as "below detection limit").
    """
    conj_rate = 0.0 if np.isnan(conj_exponent) else 10 ** conj_exponent

    # Accept either a scalar donor density or an iterable of donor densities
    if np.size(B_TC) == 1:
        return conj_rate * B_WT * B_TC

    return conj_rate * B_WT * np.sum(B_TC)


def fMultistrain(t, y, params):
    """
    RHS for the multistrain ODE model.

    State vector
    ------------
    y = [S, A, B_TC(1..n), B_WT(1..n)]
      S : resource (or limiting substrate)
      A : antibiotic concentration
      B_TC : plasmid-bearing densities (one per strain)
      B_WT : plasmid-free densities (one per strain)
    """
    S, A = y[0], y[1]
    num_strains = int(len(params["strain"]) / 2)

    B_TC = y[2 : num_strains + 2]
    B_WT = y[num_strains + 2 :]

    uStot = 0.0
    dB = np.zeros(2 * num_strains)

    # Plasmid-bearing (TC) dynamics
    for i in range(num_strains):
        uSi_TC = uS(S, params["VKm"].iloc[i])
        uStot += uSi_TC * B_TC[i]

        dB_TC_growth = params["rho"].iloc[i] * uSi_TC * B_TC[i]
        dB_TC_seg = (
            params["rho"].iloc[i]
            * uSi_TC
            * B_TC[i]
            * params["seg_rate"].iloc[i]
        )

        db_MIC_TC = params["kappa"].iloc[i]
        dB_TC_kill = dB_kill(A, B_TC[i], db_MIC_TC, params.attrs["A_max"])

        # Conjugation into the TC compartment for this strain's recipient WT
        conj_perm = params["conj_rate"].iloc[i + num_strains]
        dB_TC_conj = dB_conj(B_WT[i], B_TC, conj_perm)

        dB[i] = dB_TC_growth + dB_TC_conj - dB_TC_seg - dB_TC_kill

    # Plasmid-free (WT) dynamics
    for i in range(num_strains, 2 * num_strains):
        j = i - num_strains  # strain index (0..n-1)

        uSi_WT = uS(S, params["VKm"].iloc[i])
        uStot += uSi_WT * B_WT[j]

        dB_WT_growth = params["rho"].iloc[i] * uSi_WT * B_WT[j]

        db_MIC_WT = params["kappa"].iloc[i]
        dB_WT_kill = dB_kill(A, B_WT[j], db_MIC_WT, params.attrs["A_max"])

        # WT gain from TC segregation (loss of plasmid)
        uSi_TC = uS(S, params["VKm"].iloc[j])
        dB_WT_seg = (
            params["rho"].iloc[j]
            * uSi_TC
            * B_TC[j]
            * params["seg_rate"].iloc[j]
        )

        # WT loss via conjugation (conversion to TC)
        conj_perm = params["conj_rate"].iloc[i]
        dB_WT_conj = dB_conj(B_WT[j], B_TC, conj_perm)

        dB[i] = dB_WT_growth - dB_WT_conj - dB_WT_kill + dB_WT_seg

    dS = -uStot
    dA = -A * (
        params.attrs["alphas"][0] * np.sum(B_TC)
        + params.attrs["alphas"][1] * np.sum(B_WT)
    )

    return np.concatenate(([dS], [dA], dB))


def simulate_multistrain(this_params, istrains, y0):
    """
    Integrate the multistrain ODE system for a selected strain set.

    Parameters
    ----------
    this_params : pandas.DataFrame-like
        Full parameter table (with attrs carrying global settings).
    istrains : list-like
        Indices (or identifiers) selecting the strains to simulate.
    y0 : array-like
        Initial state vector [S, A, B_TC..., B_WT...].

    Returns
    -------
    times : ndarray
        Integration time points.
    ys : ndarray
        State trajectories with shape (n_states, n_times).
    strains_params : object
        Parameter table restricted to the selected strains.
    """
    strains_params = get_selected_strains_params(this_params, istrains)

    t_span = [0, this_params.attrs["T"]]

    # Stiff solver for potentially sharp antibiotic killing terms
    sol = solve_ivp(
        fMultistrain, t_span, y0, args=(strains_params,), method="Radau"
    )

    return sol.t, sol.y, strains_params




def simulateTransfers_multistrain(model_params, istrains, E, type_experiment='invasion', verbose=False):
    """
    Simulate the plasmid transfer dynamics for a multi-strain community.

    This function simulates either a competition experiment where all strains start with equal biomass,
    or an invasion experiment where one strain initiates with plasmid and all others are plasmid-free.
    It then applies daily serial dilutions over a course of multiple days, as per the antibiotic concentration E provided.

    Parameters:
    model_params: xr.Dataset
        Dataset containing parameters for the model.
    istrains: list
        List of strains included in the simulation.
    E: list s
        List of daily antibiotic concentrations.
    type_experiment: str, optional
        Type of experiment to simulate: 'competition' or 'invasion'. Default is 'invasion'.
    verbose: bool, optional
        If True, print more information during simulation. Default is False.

    Returns:
    times_list: list
        List of arrays, each containing time points of a simulation for each day.
    ys_list: list
        List of arrays, each containing biomass densities of each strain at different time points for each day.
    strains_params_list: list
        List of arrays, each containing parameters of each strain for each day.
    """
    times_list = []
    ys_list = []
    strains_params_list = []
    num_strains = len(istrains)

    # Set initial conditions
    S0 = model_params.attrs['S0']
    B0 = model_params.attrs['B0']

    for day in range(len(E)):
        A = E[day]

        if day == 0:
            # Competition experiment
            if type_experiment=='competition':
                yi = np.concatenate(([S0], [A], B0 * np.ones(2 * num_strains) / (num_strains * 2)))

            else : # Invasion experiment
                istrains_TC0 = np.zeros(num_strains)
                istrains_WT0 = B0 * np.ones(num_strains) / num_strains
                yi = np.concatenate(([S0], [A], istrains_TC0, istrains_WT0))
                yi[2] = B0 * 1e-3  # Only one strain initiates with plasmid at 0.1%

            # Simulation for the first day
            times, ys, strains_params = simulate_multistrain(model_params, istrains, yi)

        else:
            yi = ys[:, -1] * model_params.attrs['d']  # Serial dilution
            yi[0] = model_params.attrs['S0']  # Replenish media

            # Simulations for subsequent days
            times, ys, strains_params = simulate_multistrain(model_params, istrains, yi)

        times_list.append(times)
        ys_list.append(ys)
        strains_params_list.append(strains_params)

    return times_list, ys_list, strains_params_list


def plotTransfers(model_params, istrains, t_list, ys_list, strains_params_list,  save_path=''):
    """
    Plots the simulation results, displaying the biomass densities over time.

    Parameters:
    istrains: list
        List of strains included in the simulations.
    t_list: list of lists
        Each sublist contains the time points for one simulation.
    ys_list: list of lists
        Each sublist contains the biomass densities at each time point for one simulation.
    strains_params_list: list of dicts
        Each dictionary contains the parameters for each strain for one simulation.
    save_path: str, optional
        If provided, the plot will be saved at this path. Defaults to '' (not saving the plot).

    Returns:
    None
    """
    num_strains = int(len(strains_params_list[0]['strain']) / 2)
    num_days = len(t_list)

    plt.figure(figsize=(8, 3))
    for day in range(len(t_list)):
        t = t_list[day] + day * model_params.attrs['T']  # Add 'day' to each time point
        ys = ys_list[day]
        strains_params = strains_params_list[day]

        S = ys[0]
        B_TC = ys[2:num_strains + 2]
        B_WT = ys[num_strains + 2:]

        for i in range(len(B_WT)):
            if day == 0:  # Display legend only for the first iteration
                plt.plot(t, B_WT[i], ':') #, color=cmap_strains[istrains[i]], label=f'{codes[i]}'
            else:
                plt.plot(t, B_WT[i], ':') #, color=cmap_strains[istrains[i]]

        for i in range(len(B_TC)):
            if day == 0:  # Display legend only for the first iteration
                plt.plot(t, B_TC[i], '-') #, color=cmap_strains[istrains[i]], label=f'{strain_names[istrains[i]]}'
            else:
                plt.plot(t, B_TC[i], '-') #, color=cmap_strains[istrains[i]], label=f'{strain_names[istrains[i]]}'


        #if np.any(np.array(B_WT) > 0.0) or np.any(np.array(B_TC) > 0.0):
        #  plt.yscale('log', base=10)

    plt.xlim([0, num_days*model_params.attrs['T']])
    plt.xlabel('Time (hours)', fontsize=16)
    plt.ylabel('Density (cells/ml)', fontsize=16)
    #plt.ylim([1e-2, 1.1e9])
    #plt.legend()


    ax = plt.gca()  # Get current Axes instance
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)


    if save_path:
      plt.savefig(save_path)
      print("Exporting %s"%save_path)

    plt.show()

def plotTransfersFinalPoint(model_params, istrains, t_list, ys_list, strains_params_list, save_path=""):
    """
    Plot end-of-day densities across serial transfers for each strain.

    Notes
    -----
    For each day, the plot uses the first point (day 0) and the last point
    (final state) of both plasmid-free (WT) and plasmid-bearing (TC) subpopulations.
    """
    num_strains = int(len(strains_params_list[0]["strain"]) / 2)
    num_days = len(t_list)

    plt.figure(figsize=(8, 3))
    days = np.arange(num_days + 1)

    for i in range(num_strains):
        B_WT_day = []
        B_TC_day = []

        for day in range(num_days):
            ys = ys_list[day]

            B_TC = ys[2 : num_strains + 2]
            B_WT = ys[num_strains + 2 :]

            if day == 0:
                B_WT_day.append(B_WT[i, 0])
                B_TC_day.append(B_TC[i, 0])

            B_WT_day.append(B_WT[i, -1])
            B_TC_day.append(B_TC[i, -1])

        # Plot WT (dotted) and TC (solid) trajectories for this strain
        plt.plot(days + 1, B_WT_day, ":")
        plt.plot(days + 1, B_TC_day, "-")

    plt.xlim([1, num_days])
    plt.xlabel("Time (days)")
    plt.ylabel("Final density (cells/ml)")

    # Use log scale only if there is any positive density
    if (np.array(B_WT_day) > 0.0).any() or (np.array(B_TC_day) > 0.0).any():
        plt.yscale("log", base=10)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def import_model_params(filename, expe_params):
    """
    Load strain-level model parameters from a CSV file into a DataFrame.

    Notes
    -----
    - Missing numeric fields are stored as NaN.
    - `expe_params` is attached to the returned DataFrame via `df.attrs`.
    - `seg_rate` is currently set as a fixed constant for all rows.
    """
    model_params = {
        "strain_name": [],
        "strain_color": [],
        "specie": [],
        "strain": [],
        "type": [],
        "PCN": [],
        "MIC": [],
        "conj_rate": [],
        "VKm": [],
        "rho": [],
        "seg_rate": [],
        "kappa": [],
    }

    seg_rate = 0.002

    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        _headers = next(csvreader)

        # Expected columns: idx, name, color, specie, strain, type, PCN, MIC, conj_rate, VKm, rho, seg_rate, kappa
        for row in csvreader:
            model_params["strain_name"].append(row[1])
            model_params["strain_color"].append(row[2])
            model_params["specie"].append(row[3])
            model_params["strain"].append(row[4])
            model_params["type"].append(row[5])
            model_params["PCN"].append(float(row[6]) if row[6] else np.nan)
            model_params["MIC"].append(float(row[7]) if row[7] else np.nan)
            model_params["conj_rate"].append(float(row[8]) if row[8] else np.nan)
            model_params["VKm"].append(float(row[9]) if row[9] else np.nan)
            model_params["rho"].append(float(row[10]) if row[10] else np.nan)
            model_params["seg_rate"].append(seg_rate)
            model_params["kappa"].append(float(row[12]) if row[12] else np.nan)

    df = pd.DataFrame(model_params)
    df.attrs = expe_params
    return df


def export_model_params(model_params, filename):
    """
    Export model parameters to CSV using the expected column order.

    Notes
    -----
    Assumes rows are ordered as [TC strains..., WT strains...] (or equivalent)
    and that total rows equal 2 * num_strains.
    """
    num_strains = int(len(model_params["specie"]) / 2)

    headers = [
        "",
        "name",
        "color",
        "specie",
        "strain",
        "type",
        "PCN",
        "MIC",
        "conj_rate",
        "VKm",
        "rho",
        "seg_rate",
        "kappa",
    ]

    rows = []
    for i in range(2 * num_strains):
        row = [
            f"{i+1}",
            model_params["strain_name"][i],
            model_params["strain_color"][i],
            model_params["specie"][i],
            model_params["strain"][i],
            model_params["type"][i],
            model_params["PCN"][i],
            model_params["MIC"][i],
            model_params["conj_rate"][i],
            model_params["VKm"][i],
            model_params["rho"][i],
            model_params["seg_rate"][i],
            model_params["kappa"][i],
        ]
        rows.append(row)

    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)

    print(f"Data exported to {filename}")


def display_strain_params(model_params, istrain):
    
    """
    Display the parameters for a specified strain.

    This function displays the parameters for a TC strain and its corresponding WT strain. The parameters are 
    retrieved from the model parameters DataFrame using the provided strain index.

    Parameters:
    model_params: pandas DataFrame
        DataFrame that contains the parameters for each strain.
    istrain: int
        Index of the strain. This index corresponds to a row in the model parameters DataFrame.
    """
    # Display TC strain
    print(f"TC Strain {istrain+1}:")
    display_params(model_params, istrain)

    # Display WT strain
    print(f"\nWT Strain {istrain+tot_strains+1}:")
    display_params(model_params, istrain+tot_strains)


def print_expe_params(expe_params):
    """
    Pretty-print experiment-level parameters stored in a dict-like object.
    """
    print("Experimental Parameters")
    print("-" * 48)
    print(f"\tInitial bacterial density (B0): {expe_params['B0']}")
    print(f"\tMaximum drug concentration (A_max): {expe_params['A_max']}")
    print(f"\tAntibiotic degradation rates (alphas): {expe_params['alphas']}")
    print(f"\tLength of experiment (T): {expe_params['T']}")
    print(f"\tInitial resource concentration (S0): {expe_params['S0']}")
    print(f"\tResource decay rate (d): {expe_params['d']}")
    print(f"\tExtinction threshold: {expe_params['extinction_threshold']}")
    print("-" * 48)


def calculate_MIC(results, Amax_values):
    """
    Estimate MIC from a dose-response sweep using an extinction threshold.

    Parameters
    ----------
    results : list
        Per-dose simulation outputs; each entry must support `res[-1].sum()`
        as the final total density.
    Amax_values : array-like
        Antibiotic concentrations tested, in the same order as `results`.

    Returns
    -------
    float or None
        MIC estimate (interpolated in log-density space) or None if extinction
        is never reached.

    Notes
    -----
    Uses `model_params.attrs['extinction_threshold']` as the cutoff.
    """
    extinction_threshold = model_params.attrs["extinction_threshold"]

    prev_final_total_density = None
    prev_Amax = None

    for Amax, res in zip(Amax_values, results):
        final_total_density = res[-1].sum()

        if final_total_density < extinction_threshold:
            if prev_final_total_density is None:
                return Amax

            # Interpolate between the last surviving dose and the first extinct dose
            num = np.log(extinction_threshold) - np.log(prev_final_total_density)
            den = np.log(final_total_density) - np.log(prev_final_total_density)
            return prev_Amax + (Amax - prev_Amax) * (num / den)

        prev_final_total_density = final_total_density
        prev_Amax = Amax

    return None


def display_params(model_params, idx):
    """
    Print parameters for a single row (strain) in the model parameter table.

    Notes
    -----
    Color is taken from `cmap_strains` and wrapped using `tot_strains`.
    This function expects `tot_strains` and `cmap_strains` to exist globally.
    """
    name = model_params["strain_name"][idx]

    # Map idx to the base strain color (TC/WT share the same color)
    color_idx = idx if idx < tot_strains else idx - tot_strains
    color = cmap_strains[color_idx]

    print(f"\tName: {name}")
    print(f"\tColor: {color}")
    print(f"\tSpecie: {model_params['specie'][idx]}")
    print(f"\tStrain: {model_params['strain'][idx]}")
    print(f"\tType: {model_params['type'][idx]}")
    print(f"\tPCN: {model_params['PCN'][idx]}")
    print(f"\tMIC: {model_params['MIC'][idx]}")
    print(f"\tConjugation Rate: {model_params['conj_rate'][idx]}")
    print(f"\tVKm: {model_params['VKm'][idx]}")
    print(f"\tRho: {model_params['rho'][idx]}")
    print(f"\tSegregation Rate: {model_params['seg_rate'][idx]}")
    print(f"\tKappa: {model_params['kappa'][idx]}")


def display_model_params(model_params, istrains=[]):
    """
    Display model parameters for selected bacterial strains.

    Parameters:
    model_params: dict
        The dictionary containing model parameters.
    istrains: list
        The list of indices of the strains to be displayed. If the list is empty, parameters for all strains are displayed.
    """
    table = []
    num_strains=int(len(model_params['specie'])/2)

    if not istrains:
        istrains = list(range(num_strains))

    for i in istrains: #Plasmid-Free (TC)
        name = model_params['strain_name'][i]
        strain = model_params['strain'][i]
        specie = model_params['specie'][i]
        ptype = model_params['type'][i]
        pcn = model_params['PCN'][i]
        mic = model_params['MIC'][i]
        conj = model_params['conj_rate'][i]
        VKm = model_params['VKm'][i]
        rho = model_params['rho'][i]
        seg = model_params['seg_rate'][i]
        kappa = model_params['kappa'][i]
        row = [f"{i+1}", name, strain, specie,  ptype, pcn, mic, conj, VKm, rho, seg, kappa]
        table.append(row)

    for i in istrains: #Plasmid-bearing (WT)
        name = model_params['strain_name'][i+num_strains]
        strain = model_params['strain'][i+num_strains]
        specie = model_params['specie'][i+num_strains]
        ptype = model_params['type'][i+num_strains]
        pcn = model_params['PCN'][i+num_strains]
        mic = model_params['MIC'][i+num_strains]
        conj = model_params['conj_rate'][i+num_strains]
        VKm = model_params['VKm'][i+num_strains]
        rho = model_params['rho'][i+num_strains]
        seg = model_params['seg_rate'][i+num_strains]
        kappa = model_params['kappa'][i+num_strains]
        row = [f"{num_strains+i+1}", name, strain, specie,  ptype, pcn, mic, conj, VKm, rho, seg, kappa]
        table.append(row)

    headers = ["", "name", "strain", "specie",  "type", "PCN", "MIC", "conj_rate", "VKm", "rho", "seg_rate",  "kappa"]
    print(tabulate(table, headers, tablefmt="fancy_grid"))
def display_model_params_stats(model_params, istrains):
    """
    Print summary statistics of selected model parameters by (specie, type).

    Notes
    -----
    - If `istrains` is empty/None, uses all base strains (0..tot_strains-1).
    - Reports mean, min/max (as range), and N for each parameter.
    - Assumes the parameter table is ordered as [TC block, WT block].
    """
    headers = [
        "strain_name", "strain", "specie", "type",
        "PCN", "MIC", "conj_rate", "VKm", "rho", "seg_rate", "kappa",
    ]

    tot_strains = int(len(model_params["specie"]) / 2)
    if not istrains:
        istrains = list(range(tot_strains))

    # Build a table for both TC and WT rows corresponding to the selected strains
    table = []
    for i in istrains:
        table.append({k: model_params[k][i] for k in headers})
    for i in istrains:
        table.append({k: model_params[k][i + tot_strains] for k in headers})

    df = pd.DataFrame(table)

    stats_columns = ["conj_rate", "VKm", "rho", "seg_rate", "kappa"]
    df = df[["specie", "type"] + stats_columns]

    stats_table = []
    for (specie, ptype), group in df.groupby(["specie", "type"]):
        desc = group[stats_columns].describe()

        min_vals = desc.loc["min"]
        max_vals = desc.loc["max"]
        mean_vals = desc.loc["mean"]
        n = len(group)

        for col in stats_columns:
            stats_table.append([
                f"{specie}-{ptype}",
                col,
                mean_vals[col],
                (min_vals[col], max_vals[col]),
                n,
            ])

        # Spacer row for readability
        stats_table.append(["", "", "", "", ""])

    print(tabulate(stats_table, ["Group", "Parameter", "Mean", "Range", "N"], tablefmt="fancy_grid"))


def uS(S, VKm):
    """
    Resource uptake term.

    Notes
    -----
    VKm acts as an uptake efficiency/scaling parameter.
    """
    return (S * VKm) / (1 + S)


def dB_seg(B_TC, seg_rate, uSi_TC):
    """
    Segregational loss term for plasmid-bearing cells.

    Notes
    -----
    If `seg_rate` is NaN (missing), it is treated as 0.
    """
    if np.isnan(seg_rate):
        seg_rate = 0.0
    return seg_rate * uSi_TC * B_TC


def dB_conj_single(B_WT, B_TC, conj_exponent):
    """
    Conjugation term for one recipient-donor pair.

    Notes
    -----
    `conj_exponent` is interpreted as log10(rate). If NaN, rate is set to 0.
    """
    conj_rate = 0.0 if np.isnan(conj_exponent) else 10 ** conj_exponent
    return conj_rate * B_WT * B_TC


def dB_kill(A, Bs, kappa, A_max):
    """
    Antibiotic killing term.

    Notes
    -----
    Uses a linear-in-A killing rate scaled as 1/(kappa * A_max).
    """
    kill_rate = 1 / (kappa * A_max)
    return kill_rate * A * Bs
def fsinglestrain(t, y, params):
    """
    RHS for the single-strain ODE model with plasmid-bearing (TC) and plasmid-free (WT) subpopulations.

    State vector
    ------------
    y = [S, A, B_TC, B_WT]
      S    : resource concentration
      A    : antibiotic concentration
      B_TC : plasmid-bearing density
      B_WT : plasmid-free density
    """
    S, A, B_TC, B_WT = y
    A_max = params.attrs["A_max"]

    # TC dynamics
    uSi_TC = uS(S, params["VKm"][0])
    uStot_TC = uSi_TC * B_TC

    dB_TC_growth = params["rho"][0] * uSi_TC * B_TC
    dB_TC_seg = dB_seg(B_TC, params["seg_rate"][0], uSi_TC)

    db_MIC_TC = params["kappa"][0]
    dB_TC_kill = dB_kill(A, B_TC, db_MIC_TC, A_max)

    conj_perm = params["conj_rate"][1]  # WT permissiveness (log10 scale)
    dB_TC_conj = dB_conj_single(B_WT, B_TC, conj_perm)

    dB_TC = dB_TC_growth + dB_TC_conj - dB_TC_seg - dB_TC_kill

    # WT dynamics
    uSi_WT = uS(S, params["VKm"][1])
    uStot_WT = uSi_WT * B_WT

    dB_WT_growth = params["rho"][1] * uSi_WT * B_WT

    db_MIC_WT = params["kappa"][1]
    dB_WT_kill = dB_kill(A, B_WT, db_MIC_WT, A_max)

    dB_WT_seg = params["rho"][0] * uSi_TC * B_TC * params["seg_rate"][0]
    dB_WT_conj = dB_conj_single(B_WT, B_TC, conj_perm)

    dB_WT = dB_WT_growth - dB_WT_conj - dB_WT_kill + dB_WT_seg

    dS = -(uStot_TC + uStot_WT)
    dA = -A * (params.attrs["alphas"][0] * B_TC + params.attrs["alphas"][1] * B_WT)

    return np.array([dS, dA, dB_TC, dB_WT])


def simulate_model(model_params, y0):
    """
    Integrate the single-strain ODE system over one experiment duration.

    Parameters
    ----------
    model_params : object
        Parameter table with `attrs['T']` defining the end time.
    y0 : array-like
        Initial state [S, A, B_TC, B_WT].

    Returns
    -------
    times : ndarray
        Integration time points.
    ys : ndarray
        State trajectories with shape (n_states, n_times).
    """
    t_span = [0, model_params.attrs["T"]]

    sol = solve_ivp(
        fsinglestrain,
        t_span,
        y0,
        args=(model_params,),
        method="BDF",
        max_step=0.1,
        rtol=1e-5,
        atol=1e-8,
    )

    return sol.t, sol.y


def plot_simulation(t, ys):
    """
    Plot resource/antibiotic, absolute densities, and relative abundances over time.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    S, A, B_TC, B_WT = ys[0], ys[1], ys[2], ys[3]
    B_total = B_TC + B_WT

    # Resource and antibiotic
    axs[0].plot(t, S, label="Resource")
    axs[0].plot(t, A, label="Antibiotic (units of MIC)")
    axs[0].set_xlabel("Time (hours)", fontsize=12)
    axs[0].set_ylabel("Concentration", fontsize=12)
    axs[0].set_ylim([-0.05, 1.1])
    axs[0].legend()

    # Absolute densities
    axs[1].plot(t, B_TC, "-", label="TC", color="k")
    axs[1].plot(t, B_WT, ":", label="WT", color="k")
    axs[1].set_xlabel("Time (hours)", fontsize=12)
    axs[1].set_ylabel("Density (cells/ml)", fontsize=12)

    # Relative abundance
    rel_B_TC = B_TC / B_total
    rel_B_WT = B_WT / B_total

    axs[2].plot(t, rel_B_TC, "-", label="TC", color="k")
    axs[2].plot(t, rel_B_WT, ":", label="WT", color="k")
    axs[2].set_xlabel("Time (hours)", fontsize=12)
    axs[2].set_ylabel("Relative Abundance", fontsize=12)
    axs[2].legend()

    return fig, axs

def get_selected_strains_params(model_params, istrains):
    num_strains = int(len(model_params) / 2)

    indices_tc = model_params.loc[(model_params.index.isin(istrains)) & (model_params['type'] == 'TC')].index
    indices_wt = indices_tc + num_strains

    output = pd.DataFrame()

    for key in model_params.columns:
        if key == 'type':
            output[key] = ['TC'] * len(indices_tc) + ['WT'] * len(indices_wt)
        else:
            output[key] = model_params.loc[indices_tc, key].tolist() + model_params.loc[indices_wt, key].tolist()

    output.attrs = model_params.attrs
    return output

  
def get_species_from_istrains(model_params, istrains):
    """
    Retrieves the species names for a given list of strains.

    Parameters:
    model_params: xr.Dataset
        Dataset containing model parameters.
    istrains: list
        List of strains for which species names are to be retrieved.

    Returns:
    species: list
        List of species names corresponding to the input strains.
    """
    species = []

    for istrain in istrains:
        # Find the corresponding type in model_params and append it to the list
        this_species = model_params.loc[istrain, 'specie']
        species.append(this_species)

    # Return the list of types
    return species

def analyze_simulation(model_params, this_istrains, final_ys):
    """
    Analyzes the results of a simulation by computing total biomass and plasmid frequencies per strain per day.

    Parameters:
    model_params: xr.Dataset
        Dataset containing model parameters.
    this_istrains: list
        List of strains included in the simulation.
    final_ys: array
        Array containing the final biomass densities of each strain for each day.

    Returns:
    Btot, BpE, BpK, BfE, BfK, freqpE, freqpK: arrays
        Arrays containing the total biomass and plasmid frequencies for E. coli and Klebsiella strains for each day.
    """

    B = np.array(final_ys)
    B=B[:,2:]
    num_days = B.shape[0]
    num_strains = B.shape[1] // 2


    species=get_species_from_istrains(model_params, this_istrains)

    # Iterate over strains and add their populations to the relevant counters
    Btot=np.zeros(num_days)
    BpE=np.zeros(num_days)
    BpK=np.zeros(num_days)
    BfE=np.zeros(num_days)
    BfK=np.zeros(num_days)
    freqpE=np.zeros(num_days)
    freqpK=np.zeros(num_days)
    for day in range(num_days):

      #print("B[",day,"]=",B[day,:])

      for i in range(num_strains):
        if species[i] == 'E':
            BpE[day] += B[day, i]
            BfE[day] += B[day, i + num_strains]
        elif species[i] == 'K':
            BpK[day] += B[day, i]
            BfK[day] += B[day, i + num_strains]

        Btot[day]+=B[day, i + num_strains]+B[day, i]

      # Compute frequencies for E and K strains separately
      if Btot[day] > 0:
          freqpE[day] = BpE[day] / Btot[day]
          freqpK[day] = BpK[day] / Btot[day]
      else:
          freqpE[day] = np.nan
          freqpK[day] = np.nan

    return Btot, BpE, BpK, BfE, BfK, freqpE, freqpK


def load_simulation_results(filename):
    """
    Loads simulation results stored in a pickle file.

    Parameters:
    filename: str
        Name of the pickle file containing the simulation results.

    Returns:
    Lists containing the strains, antibiotic concentrations, biomass densities, plasmid frequencies, times,
    biomass densities at each time point, and parameters for each simulation.
    """
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results["istrains"], results["Es"],  results["Btot"], results["BpEs"], results["BpKs"], results["BfEs"], results["BfKs"], results["freqpEs"], results["freqpKs"], results["ts"], results["ys"], results["params"]

def save_simulation_results(filename, istrains, Es, Btot, BpEs, BpKs, BfEs, BfKs, freqpEs, freqpKs, ts, ys, params_list):

    """
    Saves the simulation results in a pickle file.

    Parameters:
    filename: str
        Name of the pickle file to save the simulation results in.
    istrains, Es, Btot, BpEs, BpKs, BfEs, BfKs, freqpEs, freqpKs, ts, ys, params_list: lists
        Lists containing the strains, antibiotic concentrations, biomass densities, plasmid frequencies, times,
        biomass densities at each time point, and parameters for each simulation.
    """
    results = {
        "istrains": istrains,
        "Es": Es,
        "Btot": Btot,
        "BpEs": BpEs,
        "BpKs": BpKs,
        "BfEs": BfEs,
        "BfKs": BfKs,
        "freqpEs": freqpEs,
        "freqpKs": freqpKs,
        "ts":ts,
        "ys":ys,
        "params":params_list,
    }
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def simulate_environment_multistrain(model_params, istrains, E, type_experiment):
    """
    Simulates multiple multi-strain environments and analyzes the results.

    Parameters:
    model_params: xr.Dataset
        Dataset containing parameters for the model.
    istrains: list
        List of strains included in the simulation.
    Es: list
        List of lists, each containing daily antibiotic concentrations.
    type_experiment: str
        Type of experiment to simulate: 'competition' or 'invasion'.

    Returns:
    Btots, BpEs, BpKs, BfEs, BfKs, freqpEs, freqpKs, ts, ys, params: lists
        Lists containing the total biomass, biomass densities, plasmid frequencies, time points,
        biomass densities at each time point, and parameters for each strain for each day for each simulation.
    """
    print('.', end="", flush=True)

    # Simulate transfers
    times_list, ys_list, strains_params_list = simulateTransfers_multistrain(model_params, istrains, E, type_experiment)

    # Get final points
    final_times, final_ys = get_final_points(times_list, ys_list)

    # Analyze simulation results
    Btot, BpE, BpK, BfE, BfK, freqpE, freqpK  = analyze_simulation(model_params, istrains, final_ys)

    return Btot, BpE, BpK, BfE, BfK, freqpE, freqpK, times_list, ys_list, strains_params_list



def simulate_environments_multistrain(model_params, istrains, Es, type_experiment):
    """
    Simulates multiple multi-strain environments and analyzes the results.

    Parameters:
    model_params: xr.Dataset
        Dataset containing parameters for the model.
    istrains: list
        List of strains included in the simulations.
    Es: list of lists
        Each sublist contains daily antibiotic concentrations.
    type_experiment: str
        Type of experiment to simulate: 'competition' or 'invasion'.

    Returns:
    Btots: list of arrays
        Each array contains the total biomass for each day.
    BpEs: list of arrays
        Each array contains the biomass densities of E. coli strains carrying the plasmid for each day.
    BpKs: list of arrays
        Each array contains the biomass densities of Klebsiella strains carrying the plasmid for each day.
    BfEs: list of arrays
        Each array contains the biomass densities of E. coli strains free of the plasmid for each day.
    BfKs: list of arrays
        Each array contains the biomass densities of Klebsiella strains free of the plasmid for each day.
    freqpEs: list of arrays
        Each array contains the frequencies of E. coli strains carrying the plasmid for each day.
    freqpKs: list of arrays
        Each array contains the frequencies of Klebsiella strains carrying the plasmid for each day.
    ts: list of lists
        Each sublist contains the time points.
    ys: list of lists
        Each sublist contains the biomass densities at each time point.
    params: list of lists
        Each sublist contains the parameters for each strain.
    """
    freqpEs = []
    freqpKs = []
    BpEs=[]
    BpKs=[]
    BfEs=[]
    BfKs=[]
    Btots=[]
    ts=[]
    ys=[]
    params=[]
    for E in Es:
        #print("E=%s"%E)
        Btot, BpE, BpK, BfE, BfK, freqpE, freqpK, times_list, ys_list, params_list = simulate_environment_multistrain(model_params, istrains, E, type_experiment)
        freqpEs.append(freqpE)
        freqpKs.append(freqpK)
        BfEs.append(BfE)
        BfKs.append(BfK)
        BpEs.append(BpE)
        BpKs.append(BpK)
        Btots.append(Btot)
        ts.append(times_list)
        ys.append(ys_list)
        params.append(params_list)

    return Btots, BpEs, BpKs, BfEs, BfKs, freqpEs, freqpKs, ts, ys, params


def plotTransfers(model_params, istrains, t_list, ys_list, strains_params_list,  save_path=''):
    """
    Plots the simulation results, displaying the biomass densities over time.

    Parameters:
    istrains: list
        List of strains included in the simulations.
    t_list: list of lists
        Each sublist contains the time points for one simulation.
    ys_list: list of lists
        Each sublist contains the biomass densities at each time point for one simulation.
    strains_params_list: list of dicts
        Each dictionary contains the parameters for each strain for one simulation.
    save_path: str, optional
        If provided, the plot will be saved at this path. Defaults to '' (not saving the plot).

    Returns:
    None
    """
    num_strains = int(len(strains_params_list[0]['strain']) / 2)
    num_days = len(t_list)
    cmap_strains = cc.glasbey_light[:num_strains*2]

    plt.figure(figsize=(8, 3))
    for day in range(len(t_list)):
        t = t_list[day] + day * model_params.attrs['T']  # Add 'day' to each time point
        ys = ys_list[day]
        strains_params = strains_params_list[day]

        S = ys[0]
        B_TC = ys[2:num_strains + 2]
        B_WT = ys[num_strains + 2:]

        for i in range(len(B_WT)):
            if day == 0:  # Display legend only for the first iteration
                plt.plot(t, B_WT[i], ':') #, color=cmap_strains[istrains[i]], label=f'{codes[i]}'
            else:
                plt.plot(t, B_WT[i], ':') #, color=cmap_strains[istrains[i]]


        print("*%s %s %s %s"%(i, len(B_TC), len(B_WT), len(cmap_strains)))
        for i in range(len(B_TC)):
            if day == 0:  # Display legend only for the first iteration
                plt.plot(t, B_TC[i], '-') #, color=cmap_strains[istrains[i]], label=f'{strain_names[istrains[i]]}', label=f'{codes[i]} (TC)'
            else:
                plt.plot(t, B_TC[i], '-') #, color=cmap_strains[istrains[i]], label=f'{strain_names[istrains[i]]}'


        #if np.any(np.array(B_WT) > 0.0) or np.any(np.array(B_TC) > 0.0):
        #  plt.yscale('log', base=10)

    plt.xlim([0, num_days*model_params.attrs['T']])
    plt.xlabel('Time (hours)', fontsize=16)
    plt.ylabel('Density (cells/ml)', fontsize=16)
    #plt.ylim([1e-2, 1.1e9])
    #plt.legend()


    ax = plt.gca()  # Get current Axes instance
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)


    if save_path:
      plt.savefig(save_path)
      print("Exporting %s"%save_path)

    plt.show()

def get_final_points(times_list, ys_list):
    """
    Extracts the final time point and the final solution vector for each day of simulation.

    Parameters:
    times_list: list of lists
        Each sublist contains the time points for one simulation.
    ys_list: list of lists
        Each sublist contains the biomass densities at each time point for one simulation.

    Returns:
    final_times: list
        List of final time points for each simulation day.
    final_ys: list of lists
        Each sublist contains the final solution vector for each simulation day.
    """
    final_times = []
    final_ys = []

    for day in range(len(times_list)):
        t = times_list[day]
        ys = ys_list[day]
        final_time = t[-1]  # Get the last time point
        final_y = ys[:, -1]  # Get the last solution vector
        final_times.append(final_time)
        final_ys.append(final_y)

    return final_times, final_ys

def plotTransfersFinalPoint(model_params, istrains, t_list, ys_list, strains_params_list, save_path=''):
    """
    Plots the final density of each strain at the end of each day of simulation.

    Parameters:
    istrains: list
        List of strains included in the simulations.
    t_list: list of lists
        Each sublist contains the time points for one simulation.
    ys_list: list of lists
        Each sublist contains the biomass densities at each time point for one simulation.
    strains_params_list: list of dicts
        Each dictionary contains the parameters for each strain for one simulation.
    save_path: str, optional
        If provided, the plot will be saved at this path. Defaults to '' (not saving the plot).

    Returns:
    None
    """

    num_strains = int(len(strains_params_list[0]['strain']) / 2)
    num_days = len(t_list)
    
    
    cmap_strains = cc.glasbey_light[:num_strains*2]

    plt.figure(figsize=(8, 3))
    t = np.arange(num_days)

    for i in range(num_strains):
        B_WT_day = []
        B_TC_day = []
        for day in range(num_days):
            ys = ys_list[day]
            strains_params = strains_params_list[day]

            B_TC = ys[2:num_strains + 2]
            B_WT = ys[num_strains + 2:]

            #if day==0: #Initial conditions
            #  B_WT_day.append(B_WT[i, 0])
            #  B_TC_day.append(B_TC[i, 0])

            B_WT_day.append(B_WT[i, -1])
            B_TC_day.append(B_TC[i, -1])

            #print(B_TC)


        if max(B_WT_day) > 1e7 or max(B_TC_day) > 1e7: #extinction threshold
            plt.plot(t+1, B_WT_day, ':') #, color=cmap_strains[istrains[i]]
            plt.plot(t+1, B_TC_day, '-') #, color=cmap_strains[istrains[i]], label=f'{strain_names[istrains[i]]}'
        else:
            plt.plot(t+1, B_WT_day, ':', color='grey', alpha=0.5)
            plt.plot(t+1, B_TC_day, '-', color='grey', alpha=0.5)



    plt.xlim([1, num_days+1])
    plt.xlabel('Time (days)', fontsize=16)
    plt.ylabel('Final density (cells/ml)', fontsize=16)
    plt.ylim([1e-2, 5e9])

    if np.any(np.array(B_WT_day) > 0.0) or np.any(np.array(B_TC_day) > 0.0):
        plt.yscale('log', base=10)

    ax = plt.gca()  # Get current Axes instance
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)


    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    #if save_path:
    #  plt.savefig(save_path)
    #  print("Exporting %s"%save_path)
    plt.show()

def load_environment(str_E, indx_E, envPath, num_days):
    """
    Loads the environmental data from a specified CSV file.

    Parameters:
    str_E: str
        The name of the environment.
    indx_E: int
        The index of the environment.
    envPath: str
        The directory path where the CSV files are located.
    num_days: int
        The number of days for which the environmental data is to be loaded.

    Returns:
    E: array
        The environmental data for the specified number of days.
    """

    # Concatenate the environment name and index to form the filename
    this_csv = f'{str_E}_{indx_E}.csv'

    # Join the directory path and filename to form the complete file path
    path_csv = os.path.join(envPath, str_E)

    # Load the CSV file as a pandas DataFrame
    T = pd.read_csv(os.path.join(path_csv, this_csv))

    # Extract the environmental data and normalize it to range between 0 and 1
    E_sample = T.iloc[:, 0].values
    E = (E_sample - E_sample.min()) / (E_sample.max() - E_sample.min())

    # Return the environmental data for the specified number of days
    return E[:num_days]


def load_environments(str_E, envPath, num_days, iEs):
    """
    Loads the environmental data for multiple environments.

    Parameters:
    str_E: str
        The name of the environment.
    envPath: str
        The directory path where the CSV files are located.
    num_days: int
        The number of days for which the environmental data is to be loaded.
    iEs: list
        The list of environment indices.

    Returns:
    Es: list
        The list of environmental data for each specified environment.
    """

    Es = []
    for indx_E in iEs:
        # Load the environmental data for the given index
        E = load_environment(str_E, indx_E, envPath, num_days)

        # Add the environmental data to the list
        Es.append(E)

    return Es


def plot_environment(E, Emax=1, str_E=''):
    """
    Plots the environmental data as a heatmap.

    Parameters:
    E: array
        The environmental data to be plotted.
    Emax: float, optional
        The maximum possible value of the environmental data. The colorbar will range from 0 to Emax. Defaults to 1.
    str_E: str, optional
        The title of the plot. Defaults to an empty string.
    """

    # Create a colormap and normalize the colors
    cmap = plt.get_cmap('gray_r')
    norm = mcolors.Normalize(vmin=0, vmax=Emax)

    # Create the figure and axes
    plt.figure(figsize=(10, 1))

    # Create the heatmap. The 'extent' parameter ensures that the heatmap fills the entire plot area.
    plt.imshow([E], cmap=cmap, norm=norm, aspect='auto', extent=[0, len(E)+1, 0, 1])

    # Position the x-ticks to correspond to each day, located at the center of the heatmap cells
    plt.xticks(np.arange(0, len(E) + 1)+0.5, np.arange(0, len(E) + 1))

    # Labeling, title and display
    plt.xlabel('Time (days)')
    plt.title(str_E)
    plt.show()


