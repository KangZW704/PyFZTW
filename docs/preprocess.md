# PyFZTW preprocessing and detection

The workflow in `prep/` converts horizontal north and east recordings to fault-parallel displacement and screens events for inversion using amplitude patterns and waveform template correlation.
The workflow follows Qiu et al. (2021). The Ridgecrest example in this reviewer package starts from processed displacement traces. For the SAC preparation workflow below, provide your own SAC files and four-column manifest.

```text
N/E SAC
  -> prep/step1_load_sac.py
  -> *_fpara.pkl
  -> prep/step2_detect_fztw.py
  -> amp_candidates.txt
  -> prep/step3_prepare_inversion_model.py
  -> stations.txt + crust.txt + eqs.txt + DISP/*.disp
  -> PyFZTW inversion
```


## 1. Read SAC, rotate components, and pick S arrivals

Run the following script.

```text
prep/step1_load_sac.py
```

Step 1 reads paired N/E SAC records, checks their headers, rotates the components into fault-parallel and fault-normal directions, and picks the S arrival at each station. It writes one fault-parallel pickle per event.

### 1.1 Input

The input is a whitespace-delimited file with four columns.

```text
evid station north_sac east_sac
```

Example input

```text
38569591 B401 sac/38569591/B401_N.SAC sac/38569591/B401_E.SAC
38569591 B402 sac/38569591/B402_N.SAC sac/38569591/B402_E.SAC
```

### 1.2 Command

```bash
example="my_ridgecrest_inputs"

python prep/step1_load_sac.py "$example/input_file.txt" \
  --fault-angle 300 \
  --time-window -5 55 \
  --output "$example/data"
```

Command options

- `input_file` is the four-column manifest and defaults to `input_file.txt`.
- `--output` or `--output-dir` sets the output directory and defaults to the current directory.
- `--fault-angle` sets the fault-parallel azimuth and defaults to 300 degrees.
- `--time-window START END` sets the SAC time window and defaults to -5 to 55 seconds.

### 1.3 Processing

Step 1 performs the following operations for each station.

```text
Read the north and east SAC components
  -> Check headers such as npts, delta, station and event coordinates, and event depth
  -> Rotate into fault_parallel and fault_normal using fault_angle
  -> Define the S search window from hypocentral distance and a reference Vs of 3.2 km/s
  -> Band-pass filter both rotated components for picking
  -> Pick S from the horizontal envelope peak
  -> Calculate Ssnr and Vs_est = hypo_dist_km / Spick
```

If the search window contains no clear S peak, use the theoretical arrival.

```python
Spick = hypo_dist_km / 3.2
Ssnr = -1
```

A station with unreadable or unrotatable SAC data keeps its slot in the event arrays, with unavailable waveform or pick values represented by `nan`. An event with no usable waveforms produces no pickle.

### 1.4 Output

Step 1 writes one pickle per event.

```text
{event_id}_fpara.pkl
```

Example output

```text
38690767_fpara.pkl
```

The pickle contains the following top-level fields.

```python
{
    "data_fpara": list[np.ndarray],
    "stalst": np.ndarray,
    "arrdist": np.ndarray,
    "indsort": np.ndarray,
    "Spicks": np.ndarray,
    "Pick_info": np.ndarray,
    "time_axis": np.ndarray,
}
```

Fields and conventions

- Station arrays are sorted by increasing `arrdist`.
- `data_fpara` is the list of fault-parallel waveforms. Each trace has the length of `time_axis`.
- `stalst` is the station name array.
- `arrdist` is the one-dimensional distance along the array, in km.
- `indsort` maps the input order to ascending `arrdist` order.
- `Spicks` is the array of S arrival times.
- `Pick_info` has shape `(nstation, 2)` with `Vs_est` in the first column and `Ssnr` in the second.
- `time_axis` is the shared time axis for the event.

## 2. Measure amplitude profiles and screen candidate events

Run the following script.

```text
prep/step2_detect_fztw.py
```

Step 2 reads the `*_fpara.pkl` files, computes a station amplitude profile for each event, and selects events whose profiles resemble the median profile of valid events.

### 2.1 Command

```bash
example="my_ridgecrest_inputs"

python prep/step2_detect_fztw.py "$example/data" \
  --output-dir "$example/step2" \
  --min-snr 20 \
  --freqmin 1.0 \
  --freqmax 15.0 \
  --amp-cc-threshold 0.95
```

Command options

- `input_dir` contains the `*_fpara.pkl` files from Step 1.
- `--output-dir` sets the output directory and defaults to the current directory.
- `--min-snr` sets the station SNR threshold for selecting picks used to define the event-level S window. The default is 20.
- `--freqmin` and `--freqmax` set the band-pass corners and default to 1 and 15 Hz.
- `--amp-cc-threshold` sets the amplitude-profile similarity threshold and defaults to 0.95.

### 2.2 Processing

For each event, Step 2 uses S picks from high-SNR stations to define an event-level S window.

```python
ind_snr = Pick_info[:, 1] > min_snr
s_window = percentile(Spicks[ind_snr], [10, 90])
```

It then preprocesses each fault-parallel trace for detection.

```text
Remove the mean
  -> Remove the linear trend
  -> 5% taper
  -> Integrate in time to obtain displacement
  -> Detrend and taper again
  -> Apply a zero-phase band-pass filter
  -> Apply the 1/sqrt(t) convolution correction for time > 1.0 s
```

The amplitude analysis uses the following window.

```python
time >= win_min - 0.5
time <= win_max + 1.5
```

The station-level amplitude features are the following.

- `peak amplitude`
- `duration above 0.2 * event max amplitude`
- `rms amplitude`

The event amplitude profile uses peak amplitude times RMS amplitude.

```python
combined = peak * rms
normalized_combined = combined / max(abs(combined))
```

Profiles from valid events form an event-by-station matrix.

```python
res  # shape: n_event, n_station
```

The script computes the representative amplitude profile and each event's normalized dot product with it.

```python
mean_amp = median(res[valid, :], axis=0)
amp_cc = mean(mean_amp * row) / (rms(mean_amp) * rms(row))
```

Events with `amp_cc >= amp_cc_threshold` are written to the candidate table for Step 3.

### 2.3 Output

Step 2 writes these files.

```text
Amp_distribution.npz
All_events_amp_analysis.png
amp_candidates.txt
```

`Amp_distribution.npz` contains the following arrays.

```python
res
evids
stations
arrdist
mean_amp
amp_cc
amp_selected
min_snr
freqmin
freqmax
amp_cc_threshold
```

`amp_candidates.txt` lists the events that pass amplitude screening.

```text
evid    amp_cc
38569591    0.977078
...
```

## 3. Match templates, stack waveforms, and prepare inversion inputs

Run the following script.

```text
prep/step3_prepare_inversion_model.py
```

Step 3 prepares PyFZTW inputs from the Step 1 pickles and, for stacking, the Step 2 candidate table. It supports two modes.

- `--stage 0` uses the reference event alone and writes inversion inputs directly.
- `--stage 1` reads `amp_candidates.txt`, matches candidate waveforms to the template, aligns and averages the selected traces, and writes inversion inputs.

### 3.1 Stage 0 with the reference event


```bash
example="my_ridgecrest_inputs"

python prep/step3_prepare_inversion_model.py \
  --data-dir "$example/data" \
  --ref-event 38690767 \
  --station-range B416 B423 \
  --stage 0 \
  --output-dir "$example/inversion_stage0"
```

Stage 0 extracts the S template window at the specified stations and writes the `.disp` files. The selected-event table and stack comparison plot are Stage 1 outputs.

### 3.2 Stage 1 with template matching and stacking

Stage 1 takes the `amp_candidates.txt` file from Step 2.

```bash
example="my_ridgecrest_inputs"

python prep/step3_prepare_inversion_model.py "$example/step2/amp_candidates.txt" \
  --data-dir "$example/data" \
  --ref-event 38690767 \
  --station-range B416 B423 \
  --stage 1 \
  --output-dir "$example/inversion_stage1" \
  --template-cc-threshold 0.85
```

Stage 1 performs the following operations.

```text
Read the reference and candidate event pickles
  -> Extract the reference S template window for the specified station range
  -> Extract the corresponding type of S window for each candidate
  -> Compute normalized cross-correlation with the reference at each station
  -> Align candidate waveforms at the best lag
  -> Average the traces for events with template_cc > threshold
  -> Write inversion inputs
```

The reference event is always included in the stack. The default `--template-cc-threshold` is 0.85.

### 3.3 Output files

Stage 0 writes the following files.

```text
stations.txt
crust.txt
eqs.txt
DISP/*_ref_<ref_event>.disp
```

Stage 1 also writes these files.

```text
fztw_selected_events.txt
fztw_data_<ref_event>.png
```

File contents

- `stations.txt` gives station geometry as `station x y z`. Here, `x` is the distance in km from the first selected station, and `y` and `z` are zero.
- `crust.txt` is a crust and fault-zone parameter template.
- `eqs.txt` is a source parameter template.
- `DISP/*_ref_<ref_event>.disp` gives one inversion waveform per selected station.
- `fztw_selected_events.txt` lists the events included in the Stage 1 stack.
- `fztw_data_<ref_event>.png` compares the reference and stacked waveforms.

A `.disp` header has the following form.

```text
eq 38690767
station B416
window 10.831961 12.831961
maxfreq 15
accuracy 1e-05
gibbs 5
```

Two columns follow the header.

```text
time displacement
```

The template window extends approximately from -0.5 to 1.5 s relative to the template S reference. When writing `.disp`, the script adds the mean S pick of the reference event across the selected stations.

## 4. Run the inversion

After Step 3 writes the inputs, run PyFZTW.

```bash
out="my_ridgecrest_inputs/inversion_stage1"

python -m PyFZTW \
  --crust-file "$out/crust.txt" \
  --earthquake-file "$out/eqs.txt" \
  --station-file "$out/stations.txt" \
  --seismogram-files "$out"/DISP/*.disp \
  --optimizer deap \
  --optimizer-config configs/deap.toml \
  --output "$out/inversion_results.pkl"
```

Before starting the inversion, check the following.

- The Fortran forward extension `_hs4pdis` has been compiled.
- The parameter bounds in `crust.txt` and `eqs.txt` suit the analysis.
- The station range covers the fault-zone damage zone of interest.
- The `.disp` traces have suitable time windows and polarity, and contain usable signal.

## 5. Troubleshooting

- For `No *_fpara.pkl files found`, point Step 2 to the Step 1 output directory.
- For `missing required Step 2 field(s)`, regenerate the pickles with the current Step 1 script to populate `time_axis` and the other required fields.
- Events with `fewer than two S picks with SNR > ...` are skipped because Step 2 cannot define the S window.
- For `amp_candidates_file is required when --stage 1`, provide the Step 2 candidate table for Stage 1. Stage 0 uses the reference event alone.
- For `Station range endpoint(s) not found`, check that both endpoints of `--station-range` appear in the reference event's `stalst`.
- If the `.disp` window looks wrong, inspect the reference event's `Spicks` for the selected stations.
