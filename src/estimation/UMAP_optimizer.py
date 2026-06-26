### Imports
from rail.core.data import DataStore
from rail.core.stage import RailStage
from rail.core.data import PqHandle

import numpy as np
import h5py
import tables_io

from UMAPEstimator import UMAPEstimator
import optuna

import sys
data_cut = sys.argv[1]
data_cut = int(data_cut)

### Specify path to noisy catalog
noisy_catalog_path = "/pscratch/sd/s/sajkov/data/integrated_catalog_23apr26_noised_19Jun26.pq"

### Specify path to output photometry
import time
date = time.strftime('%d%b%y', time.localtime())
photoz_path = noisy_catalog_path.split('.pq')[0] + f'_UMAPphotoz_{date}.pq'

### Static parameters
training_fraction = 0.8
metric            = "manhattan_weighted_linear"
seed              = 42

### Load and split data
data = tables_io.read(noisy_catalog_path)
data = data[:data_cut]

training_indices = np.zeros(len(data), dtype = bool)
training_indices[np.random.choice(len(data), size = int(training_fraction * len(data)),
                 replace = False)] = True

bands = [key for key in data.keys() if (not key.endswith('_err')) & (key != 'Roman_F146')]
error_bands = [key for key in data.keys() if key.endswith('_err')]

validation_data = data[~training_indices]
training_data   = data[training_indices]

photometry_bands = [key for key in training_data.keys()\
                        if (not key.endswith('_err')) and (key != 'Roman_F146')]
phot_error_bands = [f"{key}_err" for key in photometry_bands]

redshift_filepath = '/pscratch/sd/s/sajkov/data/mock_catalog_Ch1_26.h5'
redshift          = h5py.File(redshift_filepath)['sfh_parameters'][:, -1]

training_redshift   = redshift[:data_cut][training_indices]
validation_redshift = redshift[:data_cut][~training_indices]

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
    
    # evaluator = DistToPointEvaluator.make_stage(
    #     name = "DistToPointEvaluator",
    #     metrics = ["cdeloss"],
    #     reference_dictionary_key = "true_z",
    #     hdf5_groupname = "",
    #     metric_integration_limits = [0, 3],
    #     dx = 0.01,
    #     output_mode = "return"
    # )
    
    # evaluation = evaluator.evaluate(estimator.get_handle('estimated_photoz_pdfs').data,
    #                                  pd.DataFrame({"true_z": validation_redshift}))
    
    # cdeloss = evaluation["summary"].read()['cdeloss'][0]
    
    return cde_loss(pz_pdfs, validation_redshift)

storage = optuna.storages.RDBStorage("sqlite:////UMAP_optimization_datacut{data_cut:n}.db")

study = optuna.create_study(
    study_name = f"UMAP_optimization_datacut{data_cut:n}",
    storage = storage,
    direction = "minimize",
    load_if_exists = True)

study.optimize(objective, n_trials = 100)