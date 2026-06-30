### Imports
from rail.core.data import DataStore
from rail.core.stage import RailStage
from rail.core.data import PqHandle

import numpy as np
import h5py
import tables_io

import os
import sys
date = sys.argv[1]
configuration = sys.argv[2]
data_cut = int(sys.argv[3])
n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 100

sys.path.insert(0, "/global/homes/s/sajkov/rail_umap/src/estimation")
from UMAPEstimator import UMAPEstimator
import optuna

### Specify path to noisy catalog
noisy_catalog_path = "/pscratch/sd/s/sajkov/data/integrated_catalog_23apr26_noised_19Jun26.pq"

### Specify path to output photometry
photoz_path = noisy_catalog_path.split('.pq')[0] + f'_UMAPphotoz_{date}.pq'

### Static parameters
training_fraction = 0.8
metric            = "manhattan_weighted_linear"
seed              = 42

### Set up random generator
rng = np.random.default_rng(seed = seed)

### Load, magnitude-limit, and split data
data = tables_io.read(noisy_catalog_path)
i_band_cut = 27.5
magnitude_cut_idx = np.where(data["LSST_i"] < i_band_cut)[0]
data_magnitude_rand_idx = rng.choice(magnitude_cut_idx, size = int(1.25 * data_cut), replace = False)
training_indices = data_magnitude_rand_idx[:data_cut]
testing_indices = data_magnitude_rand_idx[data_cut:]

LSST_bands = [f"LSST_{band}" for band in "ugrizy"]
Roman_bands = [f"Roman_{band}" for band in ["F106", "F129", "F158", "F184", "F213"]]
HSC_bands = [f"HSC_MB_{band:02}" for band in range(16)]

bands_dict = {"LSST":         LSST_bands,
              "Roman":        Roman_bands,
              "HSC":          HSC_bands,
              "LSSTRoman":    LSST_bands  + Roman_bands,
              "RomanHSC":     Roman_bands + HSC_bands,
              "LSSTRomanHSC": LSST_bands  + Roman_bands + HSC_bands}

bands = bands_dict[configuration]
error_bands = [key + "_err" for key in bands]

print("Optimizing UMAP with bands:", bands)

training_data   = data.iloc[training_indices]
validation_data = data.iloc[testing_indices]

print("Length of training data: ", len(training_data))
print("\" \"   validation data: ", len(validation_data))

photometry_bands = [key for key in training_data.keys()\
                        if (not key.endswith('_err')) and (key != 'Roman_F146')]
phot_error_bands = [f"{key}_err" for key in photometry_bands]

redshift_filepath = '/pscratch/sd/s/sajkov/data/mock_catalog_Ch1_26.h5'
with h5py.File(redshift_filepath) as f:
    redshift = f['sps_parameters'][:, -1]

training_redshift   = redshift[training_indices]
validation_redshift = redshift[testing_indices]

def cde_loss(pz_ensemble, true_z):
    
    xvals = pz_ensemble.metadata['xvals']
    yvals = pz_ensemble.objdata['yvals']
    
    pdf_at_true = np.array([np.interp(z, xvals, y)\
        for z, y in zip(true_z, yvals)])
    integral_sq = np.trapezoid(yvals**2, xvals, axis = 1)
    
    return np.mean(integral_sq - 2 * pdf_at_true)

def objective(trial):
    
    RailStage.data_store = DataStore()
    
    n_neighbors_umap = trial.suggest_int("n_neighbors_umap", 10, 200, step  = 5)
    min_dist         = trial.suggest_float("min_dist",       0,  1,   step  = 0.05)
    
    n_neighbors_knn = trial.suggest_int("n_neighbors_knn", 5, 200, step  = 5)
    metric_p_knn    = trial.suggest_int("metric_p_knn",    1, 5,   step  = 1)
    

    estimator = UMAPEstimator.make_stage(
        name = "UMAP_estimator",
        
        ambient_metric_umap = metric,
        
        n_neighbors_umap = n_neighbors_umap,
        min_dist         = min_dist,
        
        n_neighbors_knn = n_neighbors_knn,
        metric_p_knn    = metric_p_knn,
        
        seed = seed
    )

    estimator.set_data("training_photometry", data = training_data[photometry_bands])
    estimator.set_data("training_phot_error", data = training_data[phot_error_bands])
    estimator.set_data("training_redshift",   data = training_redshift)
    
    estimator.UMAP_informer()
    
    estimator.set_data("estimation_photometry", data = validation_data[photometry_bands])
    estimator.set_data("estimation_phot_error", data = validation_data[phot_error_bands])
    estimator.UMAP_estimator()
    
    pz_pdfs = estimator.get_handle("estimated_photoz_pdfs").data
    
    return cde_loss(pz_pdfs, validation_redshift)

if not os.path.exists(f"/pscratch/sd/s/sajkov/UMAP_optimization/{date}"):
    os.makedirs(f"/pscratch/sd/s/sajkov/UMAP_optimization/{date}", exist_ok=True)

storage = optuna.storages.RDBStorage(f"sqlite:////pscratch/sd/s/sajkov/UMAP_optimization/{date}/UMAP_optimization_{configuration}.db")

study = optuna.create_study(
    study_name = f"UMAP_optimization_{configuration}_datacut{data_cut:n}",
    storage = storage,
    sampler = optuna.samplers.TPESampler(seed = seed),
    direction = "minimize",
    load_if_exists = True)

study.optimize(objective, n_trials = n_trials)