# PyFZTW

PyFZTW models and inverts fault-zone trapped waves using a Python interface to the HS4PDIS Fortran solver. This reviewer package includes a synthetic inversion example, prepared Ridgecrest waveforms, and saved results for Figures 4 and 8.

## Install and build

Run these commands from the repository root. The Conda environment includes Python 3.10, the numerical and plotting dependencies, and the Fortran build tools.

```bash
conda env create -f environment.yml
conda activate pyfztw
cd PyFZTW/forward
python -m numpy.f2py -c hs4pdis.f90 -m _hs4pdis --backend meson
cd ../..
```

## Run the synthetic example

This command runs one GA inversion using the three included synthetic traces, a fixed seed, and one worker. It writes `synthetic_ga_run01.pkl`.

```bash
python -m PyFZTW --crust-file examples/syn_test/syn_crust.txt --earthquake-file examples/syn_test/syn_eq.txt --station-file examples/syn_test/syn_stations.txt --seismogram-files examples/syn_test/data/S01_new.disp examples/syn_test/data/S02_new.disp examples/syn_test/data/S03_new.disp --optimizer deap --optimizer-config configs/deap.toml --n-runs 1 --jobs 1 --seed-base 1 --output synthetic_ga.pkl
```

Use [the optimizer guide](docs/optimizer.md) for configuration and result fields. Run `python -m PyFZTW --help` to see the available options.

## Plot the saved paper results

The scripts calculate synthetic waveforms for the saved best models and use the saved event alignment. The Ridgecrest input contains eight traces (B416–B423) with a 10.90–12.90 s fitting window.

```bash
python plot_codes/reflow_ga_pso_side_by_side.py --output synthetic_ga_pso.pdf
python plot_codes/plot_result_rid_v3.py --result examples/ridgecrest_test/ga/inversion_results_v1_run022.pkl --output ridgecrest_run022.pdf
```

[Saved result files](docs/paper-result-pickles.md) lists the three pickle inputs and the scope of the plot checks. Use trusted pickle files only.

## Prepare your own data

The three scripts in `prep/` take paired horizontal SAC recordings through S picking, amplitude-based event screening, and template matching and stacking. To use them, supply your own SAC files and four-column file list. The included synthetic and Ridgecrest examples start with processed displacement files. [Preprocessing guide](docs/preprocess.md)

