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



def load_simulation_resultsBKUP(filename):
    with open(filename, "rb") as f:
        results = pickle.load(f)
    return results["istrains"], results["Es"],  results["Btot"], results["BpEs"], results["BpKs"], results["BfEs"], results["BfKs"], results["freqpEs"], results["freqpKs"], results["ts"], results["ys"], results["params"]

def save_simulation_resultsBKUP(filename, istrains, Es, Btot, BpEs, BpKs, BfEs, BfKs, freqpEs, freqpKs, ts, ys, params_list):

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

def get_species_from_istrainsBKUP(model_params, istrains):
    # Initialize the list to store the types
    species = []

    # Loop through each element in istrains
    for istrain in istrains:
        # Find the corresponding type in model_params and append it to the list
        this_species = model_params.loc[istrain, 'specie']
        species.append(this_species)

    # Return the list of types
    return species

def dB_conj(B_WT, B_TC, conj_exponent):
    if np.isnan(conj_exponent): # Below detectable limits
        conj_rate = 0
    else:
        conj_rate=10**conj_exponent
    ret = 0
    if np.size(B_TC) == 1:
        ret = conj_rate * B_WT * B_TC  # Changed Bdonor to B_TC because in this case B_TC is a float
    else:
        for Bdonor in B_TC:
            ret += conj_rate * B_WT * Bdonor
    return ret

def fMultistrain(t, y, params):
    S = y[0]
    A = y[1]
    num_strains = int(len(params['strain']) / 2)
    B_TC = y[2:num_strains + 2]
    B_WT = y[num_strains + 2:]

    uStot = 0
    dB = np.zeros(2 * num_strains)

    # For plasmid-bearing (TC)
    for i in range(num_strains):

        uSi_TC = uS(S, params['VKm'].iloc[i])
        uStot += uSi_TC * B_TC[i]

        dB_TC_growth = params['rho'].iloc[i] * uSi_TC * B_TC[i]
        dB_TC_seg = params['rho'].iloc[i] * uSi_TC * B_TC[i] * params['seg_rate'].iloc[i]

        db_MIC_TC = params['kappa'].iloc[i]
        dB_TC_kill = dB_kill(A, B_TC[i], db_MIC_TC, params.attrs['A_max'])

        conj_permissiveness=params['conj_rate'].iloc[i+num_strains] #permissiveness of WT
        dB_TC_conj = dB_conj(B_WT[i], B_TC, conj_permissiveness)


        dB[i] = dB_TC_growth + dB_TC_conj - dB_TC_seg - dB_TC_kill

    # For plasmid-free (WT)
    for i in range(num_strains, 2*num_strains):

        uSi_WT = uS(S, params['VKm'].iloc[i])
        uStot += uSi_WT * B_WT[i-num_strains]

        dB_WT_growth = params['rho'].iloc[i] * uSi_WT * B_WT[i-num_strains]

        db_MIC_WT = params['kappa'].iloc[i]
        dB_WT_kill = dB_kill(A, B_WT[i-num_strains], db_MIC_WT, params.attrs['A_max'])

        uSi_TC = uS(S, params['VKm'].iloc[i-num_strains])
        dB_WT_seg= params['rho'].iloc[i - num_strains] * uSi_TC * B_TC[i - num_strains] * params['seg_rate'].iloc[i-num_strains]

        conj_permissiveness=params['conj_rate'].iloc[i] #permissiveness of WT
        dB_WT_conj = dB_conj(B_WT[i-num_strains], B_TC, conj_permissiveness)

        dB[i] = dB_WT_growth - dB_WT_conj - dB_WT_kill + dB_WT_seg

    dS =  - uStot
    dA = -A * (params.attrs['alphas'][0] * np.sum(B_TC) + params.attrs['alphas'][1] * np.sum(B_WT))

    return np.concatenate(([dS], [dA], dB))

def load_environmentBKUP(str_E, indx_E, envPath, num_days):
    this_csv = f'{str_E}_{indx_E}.csv'
    path_csv = os.path.join(envPath, str_E)

    #print(f'Loading {this_csv}')
    T = pd.read_csv(os.path.join(path_csv, this_csv))
    E_sample = T.iloc[:, 0].values
    E = (E_sample - E_sample.min()) / (E_sample.max() - E_sample.min())
    E = E[:num_days]

    return E

# Load environments
def load_environmentsBKUP(str_E, envPath, num_days, iEs):

  Es = []
  for indx_E in iEs:
      E = load_environment(str_E, indx_E, envPath, num_days)
      Es.append(E)
  return Es


def simulate_multistrain(this_params, istrains, y0):
    strains_params=get_selected_strains_params(this_params, istrains)

    num_strains = int((len(y0)-2) / 2)
    # Set initial conditions
    S0 = this_params.attrs['S0']
    B0 = this_params.attrs['B0']

    t_span = [0, this_params.attrs['T']]

    # Solve the ODE
    sol = solve_ivp(fMultistrain, t_span, y0, args=(strains_params,), method='Radau')

    # Extract the time points and solution vectors
    times = sol.t
    ys = sol.y

    return times, ys, strains_params



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


def get_final_pointsBKUP(times_list, ys_list):
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


def get_selected_strains_paramsBKUP(model_params, istrains):
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


def get_species_from_istrainsBKUP(model_params, istrains):
    # Initialize the list to store the types
    species = []

    # Loop through each element in istrains
    for istrain in istrains:
        # Find the corresponding type in model_params and append it to the list
        this_species = model_params.loc[istrain, 'specie']
        species.append(this_species)

    # Return the list of types
    return species


def analyze_simulationBKUP(model_params, this_istrains, final_ys):

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

def load_environmentBKUP(str_E, indx_E, envPath, num_days):
    this_csv = f'{str_E}_{indx_E}.csv'
    path_csv = os.path.join(envPath, str_E)

    #print(f'Loading {this_csv}')
    T = pd.read_csv(os.path.join(path_csv, this_csv))
    E_sample = T.iloc[:, 0].values
    E = (E_sample - E_sample.min()) / (E_sample.max() - E_sample.min())
    E = E[:num_days]

    return E

# Load environments
def load_environmentsBKUP(str_E, envPath, num_days, iEs):

  Es = []
  for indx_E in iEs:
      E = load_environment(str_E, indx_E, envPath, num_days)
      Es.append(E)
  return Es



def simulate_environment_multistrainBKUP(model_params, istrains, E, type_experiment):
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



def simulate_environments_multistrainBKUP(model_params, istrains, Es, type_experiment):
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


def plot_environmentBKUP(E, Emax=1, str_E=''):
    cmap = plt.get_cmap('gray_r')  # Choose the colormap (gray)
    num_days = len(E)
    norm = mcolors.Normalize(vmin=0, vmax=Emax)  # Normalize colors to range [0, 1]

    plt.figure(figsize=(10, 1))
    plt.imshow([E], cmap=cmap, norm=norm, aspect='auto', extent=[0, len(E)+1, 0, 1])
    plt.xticks(np.arange(0, num_days + 1)+0.5, np.arange(0, num_days + 1))  # Shift x-ticks by 0.5
    plt.xlabel('Time (days)')
    plt.xlim([0, num_days+1])
    plt.ylabel('')
    plt.title(str_E)
    plt.show()



def plotTransfersBKUP(t_list, ys_list, strains_params_list, save_path=''):
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
                plt.plot(t, B_WT[i], ':', color=cmap_strains[istrains[i]]) #, label=f'{codes[i]}'
            else:
                plt.plot(t, B_WT[i], ':', color=cmap_strains[istrains[i]])

        for i in range(len(B_TC)):
            if day == 0:  # Display legend only for the first iteration
                plt.plot(t, B_TC[i], '-', color=cmap_strains[istrains[i]]) #, label=f'{codes[i]} (TC)'
            else:
                plt.plot(t, B_TC[i], '-', color=cmap_strains[istrains[i]])


        if np.any(np.array(B_WT) > 0.0) or np.any(np.array(B_TC) > 0.0):
          plt.yscale('log', base=10)

    plt.xlim([0, num_days*model_params.attrs['T']])
    plt.xlabel('Time (hours)')
    plt.ylabel('Density (cells/ml)')
    #plt.ylim([1e-2, 1.1e9])
    #plt.legend()
    

    if save_path:
      plt.savefig(save_path)

    plt.show()


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


def plotTransfersFinalPoint(model_params, istrains,t_list, ys_list, strains_params_list, save_path=''):
    num_strains = int(len(strains_params_list[0]['strain']) / 2)
    num_days = len(t_list)

    plt.figure(figsize=(8, 3))
    t = np.arange(num_days+1)

    for i in range(num_strains):
        B_WT_day = []
        B_TC_day = []
        for day in range(num_days):
            ys = ys_list[day]
            strains_params = strains_params_list[day]

            B_TC = ys[2:num_strains + 2]
            B_WT = ys[num_strains + 2:]

            if day==0: #Initial conditions
              B_WT_day.append(B_WT[i, 0])
              B_TC_day.append(B_TC[i, 0])

            B_WT_day.append(B_WT[i, -1])
            B_TC_day.append(B_TC[i, -1])

            #print(B_TC)

        # Check if the maximum density for the strain is greater than 1e6
        if max(B_WT_day) > 1.0 or max(B_TC_day) > 1.0:
            plt.plot(t+1, B_WT_day, ':') #, color=cmap_strains[istrains[i]]
            plt.plot(t+1, B_TC_day, '-') #, color=cmap_strains[istrains[i]], label=f'{strain_names[istrains[i]]}'
        else:
            plt.plot(t+1, B_WT_day, ':') #, color=cmap_strains[istrains[i]]
            plt.plot(t+1, B_TC_day, '-') #, color=cmap_strains[istrains[i]]



    plt.xlim([1, num_days])
    plt.xlabel('Time (days)')
    plt.ylabel('Final density (cells/ml)')
    #plt.ylim([1e0, 1.1e9])
    
    if np.any(np.array(B_WT_day) > 0.0) or np.any(np.array(B_TC_day) > 0.0):
        plt.yscale('log', base=10)

    ax = plt.gca()  # Get current Axes instance
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)


    if save_path:
      plt.savefig(save_path)
    plt.show()


def import_model_params(filename, expe_params):
    # Initialize an empty dictionary
    model_params = {
         "strain_name": [], "strain_color": [], "specie": [], "strain": [], "type": [],
       "PCN": [], "MIC": [], "conj_rate": [],   "VKm": [], "rho": [], "seg_rate":[], "kappa": []}

    # Open the CSV file
    with open(filename, 'r') as csvfile:
        # Use the csv reader
        csvreader = csv.reader(csvfile)

        # Ignore the header
        headers = next(csvreader)

        seg_rate=0.002

        # Read each row: name	color	specie	strain	type	PCN	MIC	conj_rate	VKm	rho	seg_rate	Kappa
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

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(model_params)

    # Assign the experimental parameters as metadata
    df.attrs = expe_params

    # Return the completed DataFrame
    return df


def export_model_params(model_params, filename):
    num_strains = int(len(model_params['specie']) / 2)

    # Prepare the header
    headers = ["", "name", "color", "specie", "strain", "type",
                "PCN", "MIC", "conj_rate", "VKm", "rho","seg_rate", "kappa"]

    # Prepare the rows
    rows = []
    for i in range(2 * num_strains):
        name = model_params['strain_name'][i]
        color = model_params['strain_color'][i]
        specie = model_params['specie'][i]
        strain = model_params['strain'][i]
        ptype = model_params['type'][i]
        pcn = model_params['PCN'][i]
        mic = model_params['MIC'][i]
        conj = model_params['conj_rate'][i]
        VKm = model_params['VKm'][i]
        rho = model_params['rho'][i]
        seg = model_params['seg_rate'][i]
        kappa = model_params['kappa'][i]
        row = [f"{i+1}", name, color, specie, strain, ptype, pcn, mic, conj, VKm, rho, seg, kappa]
        rows.append(row)

    # Write to the CSV file
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
    print("Data exported to %s"%filename)

def print_expe_params(expe_params):
    print("Experimental Parameters:")
    print("------------------------------------------------")
    print(f"\tInitial bacterial density (B0): {expe_params['B0']}")
    print(f"\tMaximum drug concentration (A_max): {expe_params['A_max']}")
    print(f"\tAntibiotic degradation rates (alphas): {expe_params['alphas']}")
    print(f"\tLength of experiment (T): {expe_params['T']}")
    print(f"\tInitial resource concentration (S0): {expe_params['S0']}")
    print(f"\tResource decay rate (d): {expe_params['d']}")
    print(f"\tExtinction threshold: {expe_params['extinction_threshold']}")
    print("------------------------------------------------")


def calculate_MIC(results, Amax_values):

    extinction_threshold = model_params.attrs['extinction_threshold']

    prev_final_total_density = None
    prev_Amax = None

    # Loop over the results for each antibiotic concentration
    for Amax, res in zip(Amax_values, results):
        # Extract the final total bacterial density
        final_total_density = res[-1].sum()

        # If the final total bacterial density is below the threshold, interpolate MIC
        if final_total_density < extinction_threshold:
            if prev_final_total_density is not None:
                # Linear interpolation formula in log space:
                # MIC = prev_Amax + (Amax - prev_Amax) * ((np.log(extinction_threshold) - np.log(prev_final_total_density)) / (np.log(final_total_density) - np.log(prev_final_total_density)))
                return prev_Amax + (Amax - prev_Amax) * ((np.log(extinction_threshold) - np.log(prev_final_total_density)) / (np.log(final_total_density) - np.log(prev_final_total_density)))
            else:
                return Amax

        prev_final_total_density = final_total_density
        prev_Amax = Amax

    # If no MIC was found (i.e., the bacteria survived all tested concentrations), return None
    return None

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

def display_params(model_params, idx):

    # Get the parameters for the specific strain
    name = model_params['strain_name'][idx]
    #color = model_params['strain_color'][idx]
    if idx<tot_strains:
      color=cmap_strains[idx]
    else:
      color=cmap_strains[idx-tot_strains]
    specie = model_params['specie'][idx]
    strain = model_params['strain'][idx]
    ptype = model_params['type'][idx]
    pcn = model_params['PCN'][idx]
    mic = model_params['MIC'][idx]
    conj = model_params['conj_rate'][idx]
    VKm = model_params['VKm'][idx]
    rho = model_params['rho'][idx]
    seg = model_params['seg_rate'][idx]
    kappa = model_params['kappa'][idx]

    # Print the parameters
    print(f"\tName: {name}")
    print(f"\tColor: {color}")
    print(f"\tSpecie: {specie}")
    print(f"\tStrain: {strain}")
    print(f"\tType: {ptype}")
    print(f"\tPCN: {pcn}")
    print(f"\tMIC: {mic}")
    print(f"\tConjugation Rate: {conj}")
    print(f"\tVKm: {VKm}")
    print(f"\tRho: {rho}")
    print(f"\tSegregation Rate: {seg}")
    print(f"\tKappa: {kappa}")

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
    table = []
    
    headers = ["strain_name", "strain", "specie", "type", "PCN", "MIC", "conj_rate", "VKm", "rho", "seg_rate", "kappa"]
    
    tot_strains=int(len(model_params['specie'])/2)
    if not istrains:
        istrains = list(range(tot_strains))

    for i in istrains: # Combine Plasmid-Free (TC) and Plasmid-bearing (WT)
        row = {}
        for key in headers:
            row[key] = model_params[key][i]
        table.append(row)
  
    for i in istrains: # Combine Plasmid-Free (TC) and Plasmid-bearing (WT)
        row = {}
        for key in headers:
            row[key] = model_params[key][i+tot_strains]
        table.append(row)
    
    df = pd.DataFrame(table)
    
    # Select only relevant columns for statistics
    stats_columns = ["conj_rate", "VKm", "rho", "seg_rate", "kappa"]
    df = df[["specie", "type"] + stats_columns]

    # Separate by E and K, and by TC and WT
    groups = df.groupby(["specie", "type"])

    stats_table = []
    for name, group in groups:
        stats = group[stats_columns].describe()
        # Extract mean and range for each column
        min_values = stats.loc['min']
        max_values = stats.loc['max']
        range_values = max_values - min_values
        mean_values = stats.loc['mean']

        # Add to table
        for column in stats_columns:
            stats_row = [f"{name[0]}-{name[1]}", column, mean_values[column], (min_values[column], max_values[column]), len(group)]
            stats_table.append(stats_row)
        
        # Add spacer for readability
        stats_table.append(['']*len(stats_table[0]))

    print(tabulate(stats_table, ["Group", "Parameter", "Mean", "Range", "N"], tablefmt="fancy_grid"))


def uS(S, VKm):
    return (S * VKm) / (1 + S)

def dB_seg(B_TC, seg_rate, uSi_TC):
    # Calculate segregational loss rate, only applicable for plasmid-bearing population
    if np.isnan(seg_rate):
        seg_rate = 0
    return seg_rate * uSi_TC * B_TC


def dB_conj_single(B_WT, B_TC, conj_exponent):
    if np.isnan(conj_exponent): # Below detectable limits
        conj_rate = 0
    else:
        conj_rate = 10**conj_exponent
    ret = conj_rate * B_WT * B_TC
    return ret

def dB_kill(A, Bs, kappa, A_max):
    kill_rate = 1 / (kappa*A_max)
    return kill_rate * A * Bs

# This function implements a set of ordinary differential equations describing
# the population dynamics of plasmid-bearing (TC) and plasmid-free (WT) cells
# competing for a limiting resource and exposed to a bactericidal antibiotic.
# State variables are Resource (S) and Antibiotic (A) concentrations,
# and Densities of Plasmid-bearing (B_TC) and Plasmid-free (B_WT) bacteria.
def fsinglestrain(t, y, params):
    S = y[0]
    A = y[1]
    B_TC = y[2]
    B_WT = y[3]

    A_max=params.attrs['A_max']

    # For plasmid-bearing (TC)
    uSi_TC = uS(S, params['VKm'][0])
    uStot_TC = uSi_TC * B_TC

    dB_TC_growth = params['rho'][0] * uSi_TC * B_TC
    dB_TC_seg = dB_seg(B_TC, params['seg_rate'][0], uSi_TC)

    # For plasmid-free (WT)
    db_MIC_TC = params['kappa'][0]
    dB_TC_kill = dB_kill(A, B_TC, db_MIC_TC, A_max)
    dB_WT_seg= dB_seg(B_TC, params['seg_rate'][0], uSi_TC)

    conj_permissiveness = params['conj_rate'][1] # permissiveness of WT
    dB_TC_conj = dB_conj_single(B_WT, B_TC, conj_permissiveness)
    #print("TC: ",dB_TC_growth, dB_TC_conj, dB_TC_seg, dB_TC_kill)
    dB_TC = dB_TC_growth + dB_TC_conj - dB_TC_seg - dB_TC_kill

    # For plasmid-free (WT)
    uSi_WT = uS(S, params['VKm'][1])
    uStot_WT = uSi_WT * B_WT

    dB_WT_growth = params['rho'][1] * uSi_WT * B_WT

    db_MIC_WT = params['kappa'][1]
    dB_WT_kill = dB_kill(A, B_WT, db_MIC_WT, A_max)

    dB_WT_seg= params['rho'][0] * uSi_TC * B_TC * params['seg_rate'][0]

    conj_permissiveness = params['conj_rate'][1] # permissiveness of WT
    dB_WT_conj = dB_conj_single(B_WT, B_TC, conj_permissiveness)
    #print("WT: ",dB_WT_growth, dB_WT_conj, dB_WT_seg, dB_WT_kill)
    
    dB_WT = dB_WT_growth - dB_WT_conj - dB_WT_kill + dB_WT_seg

    dS =  - (uStot_TC + uStot_WT)
    dA = -A * (params.attrs['alphas'][0] * B_TC + params.attrs['alphas'][1] * B_WT)

    return np.array([dS, dA, dB_TC, dB_WT])

#The function simulate_model(model_params, y0) runs the simulation of our model
#by solving the system of ordinary differential equations (ODEs).
#It uses the solve_ivp method from the SciPy library with backward differentiation,
#returning the time points and corresponding solution vectors.
def simulate_model(model_params, y0):

    t_span = [0, model_params.attrs['T']]

    # Solve the ODE
    sol = solve_ivp(fsinglestrain, t_span, y0, args=(model_params,), method='BDF', max_step=0.1, rtol=1e-5, atol=1e-8)

    # Extract the time points and solution vectors
    times = sol.t
    ys = sol.y

    return times, ys

#The plot_simulation(t, ys) function generates three plots to visualize
#the simulation results: the first plot shows the resource and antibiotic
#concentrations over time; the second plot depicts the density of the
#plasmid-bearing and plasmid-free strains over time; the third plot
#illustrates the relative abundance of each strain over time.
def plot_simulation(t, ys):
    fig, axs = plt.subplots(1, 3, figsize=(12,3))

    S = ys[0]
    A = ys[1]
    B_TC = ys[2]
    B_WT = ys[3]
    B_total=B_TC+B_WT

    axs[1].plot(t, B_TC,'-', label='TC', color='k')

    axs[1].plot(t, B_WT, ':', label='WT', color='k')

    # Plot bacterial density
    axs[0].plot(t, S, label='Resource')
    axs[0].plot(t, A, label='Antibiotic (units of MIC)')

    axs[0].set_xlabel('Time (hours)', fontsize=12)
    axs[0].set_ylabel('Concentration', fontsize=12)
    axs[0].set_ylim([-0.05,1.1])

    axs[1].set_xlabel('Time (hours)', fontsize=12)
    axs[1].set_ylabel('Density (cells/ml)', fontsize=12)

    # Calculate relative abundance of each strain
    rel_B_TC = B_TC / B_total
    rel_B_WT = B_WT / B_total

    # Plot relative abundance
    axs[2].plot(t, rel_B_TC, '-', label='TC', color='k')
    axs[2].plot(t, rel_B_WT, ':', label='WT', color='k')

    axs[2].set_xlabel('Time (hours)', fontsize=12)
    axs[2].set_ylabel('Relative Abundance', fontsize=12)

    axs[0].legend()
    axs[2].legend()

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


