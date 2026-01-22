Repository for: <br><br>
**Temporal correlations in selection pressure determine plasmid stability in stochastic environments**<br>
_Carles Tardío Pi, Adri\'an Gonz\'alez Casanova, \'Alvaro San Mill\'and Rafael Peña-Miller_._.

This repository contains Jupyter notebooks implementing and analyzing models of plasmid dynamics in bacterial communities under antibiotic exposure.

The notebooks focus on multistrain population dynamics, conjugation, segregation, and community-level outcomes such as plasmid frequency and diversity.

---

## Jupytern Notebooks

### [plasmidNoise_multistrain_model.ipynb](plasmidNoise_multistrain_model.ipynb)
Defines the multistrain plasmid model and its core components.

This notebook:
- implements the ODE model for plasmid-bearing (TC) and plasmid-free (WT) subpopulations,
- defines growth, killing, conjugation, and segregation terms,
- provides helper functions used by downstream simulations.

---

### [plasmidNoise_communities_simulation.ipynb](plasmidNoise_communities_simulation.ipynb)
Runs community-level simulations under different antibiotic environments.

This notebook:
- loads predefined environmental time series,
- runs multistrain simulations across environments and antibiotic amplitudes,
- stores time-resolved population dynamics and frequencies.

---

### [plasmidNoise_communities_analysis.ipynb](plasmidNoise_communities_analysis.ipynb)
Analyzes simulation outputs.

This notebook:
- extracts final densities and plasmid frequencies from stored data,
- computes summary statistics (means, AUCs, diversity indices),
- generates figures and aggregate metrics from the simulation results.

## Authors

[@Systems Biology Lab, CCG-UNAM](https://github.com/ccg-esb-lab)


## License

[MIT](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License - see the [license.txt](../license.txt) file for details. Hardware is lincesed under the CERN license.
