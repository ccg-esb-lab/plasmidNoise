Repository for: <br><br>
**Temporal correlations in selection pressure determine plasmid stability in stochastic environments**<br>
_Carles Tardío Pi, Adri\'an Gonz\'alez Casanova, \'Alvaro San Mill\'and Rafael Peña-Miller_.

---

## Model background

The population dynamics model implemented in this repository, as well as the parameterization to clinical isolates, are based on the framework presented in:

> **Antimicrobial resistance level and conjugation permissiveness shape plasmid distribution in clinical enterobacteria**  
> A. Alonso-del Valle, L. Toribio-Celestino, A. Quirant, C. Tardío Pi, J. DelaFuente, R. Cantón, E. Rocha, C. Úbeda, R. Peña-Miller, and A. San Millán.  
> *Proceedings of the National Academy of Sciences* **120**(51), 2023.

The scripts and data used to generate the theoretical figures in that work are available at:

https://github.com/ccg-esb/EvK

The present repository extends and adapts this modeling framework to explore multistrain community dynamics and temporally varying antibiotic environments.

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
