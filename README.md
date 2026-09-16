# Information Assessment

Tools and resources for assessing the information available in a
dataset and the prior knowledge that can be used to build **hybrid
models** for the chemical process industry (CPI). The code in this
repository implements an *InfoQ-style roadmap* applied to a
biodiesel (transesterification) reactor case study built on top of
the BDsim benchmark.

## Background

Hybrid models combine first-principles (mechanistic) knowledge with
data-driven components. Their accuracy and reliability depend on
two complementary sources of information:

1. the **data** that is available for training, and
2. the **prior knowledge** that can be embedded in the structure or
   parameters of the model.

Before committing to a given modelling strategy it is useful to
audit how much information each source actually contributes. This
repository provides reproducible scripts and helpers to do that for
a representative CPI dataset.

## Repository layout

| Path                       | Purpose |
|----------------------------|---------|
| `InfoQ_roadmap.py`         | Main assessment pipeline. Runs the full set of data-quality and prior-knowledge checks for the three training datasets. |
| `entropy_functions.py`     | Standalone helper that scores each measurement column by the log-determinant divergence of its empirical density against a uniform reference. |
| `parallel_analysis.py`     | Library of utility functions used by the pipeline: Horn's parallel analysis for PCA retention and permutation-based relevance scores (Pearson and mutual information). |
| `whitebox.py`              | First-principles biodiesel reactor model (state ODE, simulation, kinetic parameter identification). |
| `datasets/dataset1`        | Training data with a single feed-grade (operating regime). |
| `datasets/dataset2`        | Training data with three feed-grades. |
| `datasets/dataset3`        | Design-of-Experiments (DoE) training data. |
| `datasets/dataset4`        | Design-of-Experiments (DoE) test data, held out for generalisation experiments. |

Each `datasets/dataset*` folder contains:

* `csv_measurements.csv` -- time-stamped process measurements
  (`t/s`, reactor temperature, oil flow, filter pressure, methanol
  temperature, methanol flow, oil temperature, lab biodiesel mole
  fraction);
* `csv_inputs.csv`, `csv_setpoints.csv`, `csv_states.csv` --
  manipulated variables, controller set-points and detailed
  simulator states;
* `measurements_spectra_reactor.csv` -- NIR spectra (used in
  spectroscopy-related extensions);
* auxiliary artefacts (KPI/measurements/spectra plots, JMP / Excel
  exports of the DoE designs).

## Installation

The project targets Python 3.10+ and depends on:

```
numpy
pandas
scipy
scikit-learn
seaborn
matplotlib
```

Install them into your environment with

```bash
pip install numpy pandas scipy scikit-learn seaborn matplotlib
```

No `requirements.txt` is shipped; the modules import nothing exotic
beyond the scientific Python stack.

## Usage

The scripts assume the repository root is the working directory so
that the relative paths in `pd.read_csv('datasets/...')` resolve
correctly. From the repository root:

```bash
python InfoQ_roadmap.py     # full assessment for datasets 1-3
python entropy_functions.py # per-variable information content (dataset 3)
```

`InfoQ_roadmap.py` is organised as a sequence of Spyder/Jupyter cells
delimited by `#%%` markers, so it is most comfortable to open in an
IDE that respects cell boundaries (Spyder, VS Code, Jupyter).
Running it as a plain script also works -- the script finishes by
calling `plt.show()` on every figure.

## Assessment dimensions

The pipeline implements three dimensions of the InfoQ roadmap plus a
prior-knowledge check.

### D1 -- Resolution

Counts the number of samples, the number of unique values for every
measurement column, the measurement period and the average sampling
frequency. The last 12 hours of each dataset (720 samples at the
1-minute down-sampling stride used by the pipeline) are reserved for
testing, so the figures shown are computed on the training slice.

### D2 -- Structure

Three sub-checks are produced for each dataset:

* **Colinearity of X** -- Pearson correlation heatmap of the
  measurement matrix, followed by Horn's parallel analysis of the
  standardised features. The reported "Colinearity" percentage is
  ``100 * (1 - n_components_to_retain / n_features)``.
* **Sparsity of the X / Y relationship** -- permutation-tested
  Pearson correlation between each measurement column and the
  biodiesel mole fraction. Bars corresponding to p-values above
  0.05 are masked to zero.
* **Nonlinearity of the X / Y relationship** -- permutation-tested
  mutual information between each measurement column and the
  biodiesel mole fraction, masked at the same 0.05 threshold.

### D6 -- Generalizability

K-Means clustering of the standardised features is performed for
``k = 2, ..., 9`` and four internal validation indices are recorded:

* Sum of squared distances / inertia (lower is better, elbow method).
* Calinski-Harabasz index (higher is better).
* Davies-Bouldin index (lower is better).
* Silhouette coefficient (higher is better).

The combined picture indicates whether the operating data spans a
single mode, several distinct modes, or a continuum -- information
that directly informs how aggressively a hybrid model should
regularise.

### Prior knowledge

The first-principles reactor in `whitebox.py` is fitted to the
training slice of each dataset by non-linear least squares. The
returned kinetic parameter is then tested for statistical
significance via a t-statistic that contrasts it with the residual
standard error of the fit.

## Outputs

Running `InfoQ_roadmap.py` produces, for every dataset in scope:

* time-series plots of the biodiesel mole fraction, reactor
  temperature and oil feed flow;
* a Pearson correlation heatmap and a parallel-analysis scree plot
  with the inferred colinearity percentage;
* bar plots of the masked Pearson correlations and mutual
  information scores against the biodiesel mole fraction;
* a 2x2 grid of cluster-validation curves (inertia, CH, DB,
  silhouette).

Running `entropy_functions.py` produces a single bar plot of the
log-determinant divergence of every measurement column in
`datasets/dataset3`.

## Known issues / limitations

* `entropy_functions.py` is a script, not a library. Before this
  revision, importing it caused the demo plot to render; the demo is
  now guarded by `if __name__ == "__main__":`.
* The information-content helpers (`FeatureRelevance_MI`,
  `FeatureRelevance_Pearson`, `ParallelAnalysis`) rely on
  `np.random` without setting a seed -- results are stochastic.
* `whitebox.py` ships a placeholder kinetic law
  (`kinetics(par) -> par`) so that the identification loop can be
  exercised end-to-end. Replace with an Arrhenius expression when a
  richer description is required.
* `InfoQ_roadmap.py` references datasets 1-3; `dataset4` is provided
  as a held-out DoE test set but is not consumed by the main script.

## License

Released under the MIT License. See `LICENSE` for the full text.