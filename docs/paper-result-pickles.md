# Paper result files

The reviewer package includes the saved inversion results for Figures 4 and 8.
Paths are relative to the repository root.

| Paper figures | Files | Verification |
| --- | --- | --- |
| Fig. 4, synthetic GA | `examples/syn_test/ga/inversion_results_v1_run086.pkl` | Rendering Fig. 4 with run086 and PSO run04 produced identical page content and rendered pixels to the submitted PDF. |
| Fig. 4, synthetic PSO | `examples/syn_test/pso/inversion_results_v1_run04.pkl` | Verified against the submitted Fig. 4 together with GA run086. |
| Fig. 8, Ridgecrest GA | `examples/ridgecrest_test/ga/inversion_results_v1_run022.pkl` | The local run022 reproduces the saved fitness (0.8828011549) and event alignment when evaluated on the prepared B416–B423 waveforms. |

The three pickles total about 16 MiB. Git ignore exceptions name these files individually.
The full GA ensembles and the grid behind Fig. 7 remain local.

From the repository root, recreate the plots from the saved results:

```bash
python plot_codes/reflow_ga_pso_side_by_side.py --output synthetic_ga_pso.pdf
python plot_codes/plot_result_rid_v3.py --result examples/ridgecrest_test/ga/inversion_results_v1_run022.pkl --output ridgecrest_run022.pdf
```
