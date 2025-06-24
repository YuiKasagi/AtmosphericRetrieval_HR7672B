# AtmosphericRetrieval_HR7672B

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15702022.svg)](https://doi.org/10.5281/zenodo.15702022)

This repository contains the retrieval analysis of the L dwarf HR 7672B observed with REACH/Subaru, as presented in Kasagi et al. (2025).

The 1D spectra are available in the `data/` directory.

The data reduction of REACH data was performed using [PyIRD](https://github.com/prvjapan/pyird).

This code works with `exojax` version 1.5.1.
You can install the required packages using the `requirements.txt` file by running:
```
pip install -r requirements.txt
```

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
Modify `setting.py` to set paths for:
- The observed spectrum
- The molecular database
- The output directory
- The telluric spectrum

Then, modify `main_hmc.py` to adjust:
- Molecular species
- Initial parameters
- Output directory
- Warmup/sample numbers for HMC, etc.

### 2. Run MCMC
Execute `main_hmc.py` with the `--run` option. The samples will be saved, dividing the total sample numbers by the iteration number.

```bash
python main_hmc.py --run
```

Analysis flow in `main_hmc.py`:
1. Settings -- Configure parameters
2. Read Files -- Load observation data
3. Opacity -- Define opacity settings 
4. Models -- Define models
5. Optimization (currently optional) -- Perform optimization
6. HMC -- Run HMC

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

## Citation

If you use this code or data in your work, please cite the following Zenodo record:

Kasagi, Yui. (2025). *Atmospheric retrieval of HR 7672 B: Python code and input data* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.15702022