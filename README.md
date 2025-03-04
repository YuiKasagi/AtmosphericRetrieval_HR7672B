# AtmosphericRetrieval_HR7672B

This repository contains the retrieval analysis of the L dwarf HR 7672B observed with REACH/Subaru, as presented in Kasagi et al. (2025).

The 1D spectra are available in the `data/` directory.

The data reduction of REACH data was performed using [PyIRD](https://github.com/prvjapan/pyird).

## Files

- `main_hmc.py`: Main code for running Hamiltonian Monte Carlo (HMC). The model fit is performed using both high-resolution spectra and broad-band magnitude.
    ### Options:
    - `-o, --order`: Specify echelle orders to be analyzed.
    - `-t, --target`: Only HR7672B is accepted for now.
    - `-d, --date`: Specify the observation date for setting the file name.
    - `--mmf`: Specify mmf1 or mmf2.
    - `--fit_cloud`: Use the model including cloud opacity.
    - `--fit_speckle`: Use the model including speckle (scaled host star's observed spectrum).
    - `--run`: Run MCMC or just save predictions from posterior.

- `plothmc_models.py`: Creates figures from predictions (Figure 6 in Kasagi et al. 2025).

## Usage

### 1. Change Settings
Modify `main_hmc.py` to adjust:
- Molecular species
- Initial parameters
- Output directory
- Warmup/sample numbers for HMC, etc.

### 2. Run MCMC
Execute `main_hmc.py` with the `--run` option. The samples will be saved, dividing the total sample numbers by the iteration number.

```bash
python main_hmc.py --run
```

### 3. Connect Output Samples
Once the MCMC has finished, modify main_hmc.py to specify the path to the created sample file and connect the output samples.

### 4. Generate Predictions
Execute main_hmc.py without the --run option. Predictions and models created by the median values of the posteriors will be saved.

```bash
python main_hmc.py
```

### 5. Plot Models
Finally, execute plothmc_models.py to generate the model plots.

```bash
python plothmc_models.py
```