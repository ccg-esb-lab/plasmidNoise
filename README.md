Repository for: <br><br>
**Temporal correlations in selection pressure determine plasmid stability in stochastic environments**<br>
*C. Tardio Pi, A. González Casanova, A. San Millán and R. Peña-Miller*.

---

## Model overview

This repository implements a population-dynamics model for plasmid maintenance in bacterial populations exposed to temporally variable antibiotic selection. Each strain is represented by two subpopulations: plasmid-bearing cells, (B_p), and plasmid-free cells, (B_\emptyset). Plasmid-bearing cells can have reduced antibiotic-induced killing, but may pay a growth cost. Plasmid-free cells can arise through segregational loss, and plasmids can spread back into plasmid-free cells through conjugation. The model is simulated as a serial-transfer experiment, where bacterial densities are propagated through repeated growth-dilution cycles under defined antibiotic environments. These environments can be constant, periodic, or stochastic, allowing the simulations to test how the temporal structure of selection affects plasmid fraction, population density, persistence, and extinction.

---

## Jupyter notebooks

### [plasmidNoise_constant.ipynb](plasmidNoise_constant.ipynb)

This notebook introduces the core serial-transfer simulations and analyzes plasmid-bearing and plasmid-free population dynamics under constant antibiotic concentrations. It includes single-strain dose-response simulations and community-size simulations used to quantify how antibiotic selection affects final density and plasmid fraction.

---

### [plasmidNoise_periodic.ipynb](plasmidNoise_periodic.ipynb)

This notebook studies deterministic temporal variation in antibiotic exposure. It generates periodic treatment schedules and compares single-strain and multistrain community responses across different fluctuation periods, focusing on plasmid dynamics, extinction, persistence, and final population structure.

---

### [plasmidNoise_stochastic.ipynb](plasmidNoise_stochastic.ipynb)

This notebook analyzes stochastic antibiotic environments with different temporal correlation structures. It simulates single strains and multistrain communities across stochastic replicates and summarizes survival, persistence time, final density, plasmid fraction, and the temporal relationship between antibiotic fluctuations and plasmid dynamics.

---

## Authors

[@Systems Biology Lab, CCG-UNAM](https://github.com/ccg-esb-lab)

---

## License

[MIT](https://choosealicense.com/licenses/mit/)
