# Predictive maintenance of aircraft engine

Reference: [NASA's CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

## Project Summary

### What we built
An end-to-end Remaining Useful Life (RUL) prediction pipeline 
for aircraft turbofan engines using the NASA C-MAPSS FD001 dataset.

### Problem Statement
Predict how many operational cycles a turbofan engine has remaining 
before failure — enabling proactive maintenance scheduling and 
reducing AOG (Aircraft on Ground) risk.

### Dataset
- 100 engines run to failure in training set
- 21 sensor readings per cycle
- Single operating condition, single fault mode (HPC degradation)
- Test set: 100 engines at unknown cutoff point before failure

### Approach

**1. Exploratory Data Analysis**
- Engine lifespans range from 128 to 362 cycles, mean ~206
- 7 out of 21 sensors had near-zero variance — dropped before modeling
- Key sensors showed clear monotonic degradation trends — 
  temperature sensors rising, pressure ratio declining over engine life

**2. RUL Label Generation**
- Computed RUL as max_cycle minus current_cycle per engine
- Applied piecewise linear cap at 125 cycles — standard PHM technique
- Focuses model on degradation phase, not healthy early-life phase

**3. Feature Engineering**
- Rolling mean (window=10) per sensor per engine — captures trend
- Rolling std (window=10) per sensor per engine — captures variability
- Computed per engine using groupby to prevent cross-engine contamination

**4. Modeling
- XGBoost Regressor — gradient boosted trees
- GroupKFold cross validation (5 folds) — no engine-level data leakage
- Standard scaling fitted on training data only

### Results

| Metric | CV (5-fold) | Test Set |
|--------|-------------|----------|
| RMSE   | 16.5 cycles | 19.0 cycles |
| MAE    | -           | 13.9 cycles |

### Key Findings
- Rolling mean features dominated importance — degradation is 
  a trend problem, not a volatility problem
- Top sensors: LPT outlet temperature (sensor_4), bypass ratio 
  (sensor_15), LPC outlet temperature (sensor_2), HPC static 
  pressure (sensor_11)
- These align with physical understanding — thermodynamic and 
  mechanical efficiency loss drives turbofan degradation
- Model performs best at low RUL values — most accurate when 
  failure is imminent, which is the operationally critical window

### Limitations
- FD001 is a single operating condition dataset — 
  real fleets have heterogeneous conditions (FD002/FD004)
- No uncertainty quantification — point estimates only
- Baseline model only — TCN or LSTM would capture 
  sequence patterns more effectively

### Next Steps
- Hyperparameter tuning with Optuna
- Test on FD002/FD004 for generalization across operating conditions
- LSTM or TCN for sequence-aware modeling
- Uncertainty quantification — output RUL with confidence intervals
- SHAP values for more reliable feature importance