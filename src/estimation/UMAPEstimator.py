### Default RAIL imports
from rail.core.stage import RailStage
from rail.core.data import PqHandle, ModelHandle
from ceci.config import StageParameter

### Specific imports
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.neighbors import NearestNeighbors

from numba import njit
@njit
def manhattan_weighted_linear(vec1, vec2):
    
    n_bands = len(vec1) // 2
    
    distance = 0.0
    for i in range(n_bands):
        w         = vec1[i + n_bands]**2 + vec2[i + n_bands]**2
        distance += np.abs(vec1[i] - vec2[i])/w

    return distance

class UMAPEstimator(RailStage):
    
    ### stage name
    name = 'UMAPEstimator'
    
    ### inputs and outputs
    inputs = [("training_photometry",   PqHandle),
              ("training_phot_error",   PqHandle),
              ("training_redshift",     PqHandle),
              ("estimation_photometry", PqHandle),
              ("estimation_phot_error", PqHandle)]
    
    outputs = [("informed_reducer",     ModelHandle),
               ("informed_knn",         ModelHandle),
               ("informed_embedding",   PqHandle),
               ("estimated_embedding", PqHandle),
               ("estimated_photozs",   PqHandle)]
    
    ### configuration
    config_options = RailStage.config_options.copy()
    
    config_options.update(dict(
        # UMAP parameters
        n_neighbors_umap     = StageParameter(int,   80),
        n_components         = StageParameter(int,   3),
        ambient_metric_umap  = StageParameter(str,   'euclidean'),
        n_epochs             = StageParameter(int,   None),
        learning_rate        = StageParameter(float, 1.0),
        init                 = StageParameter(str,   'spectral'),
        min_dist             = StageParameter(float, 0.0),
        spread               = StageParameter(float, 1.0),
        low_memory           = StageParameter(bool,  True),
        set_op_mix_ratio     = StageParameter(float, 1.0),
        local_connectivity   = StageParameter(int,   1),
        repulsion_strength   = StageParameter(float, 1.0),
        negative_sample_rate = StageParameter(int,   5),
        transform_queue_size = StageParameter(float, 4.0),
        metric_kwds          = StageParameter(dict,  None),
        target_n_neighbors   = StageParameter(int,   -1),
        target_metric_umap   = StageParameter(str,   'categorical'),
        target_metric_kwds   = StageParameter(dict,  None),
        target_weight        = StageParameter(float, 0.5),
        transform_seed       = StageParameter(int,   42),
        
        # k-nearest neighbor clustering parameters
        n_neighbors_knn      = StageParameter(int, 10),
        metric_knn           = StageParameter(str, 'euclidean'),
        
        # random state
        random_state         = StageParameter(int, 42)
        ))
 
 
    def UMAP_informer(self):
        
        photometry = self.get_data("training_photometry")
        
        input_data = photometry
        
        if self.config.ambient_metric_umap == 'manhattan_weighted_linear':
            
            phot_error = self.get_data("training_phot_error")
            input_data = pd.concat([photometry, phot_error], axis = 1)
            metric = manhattan_weighted_linear
        
        else:
            metric = self.config.ambient_metric_umap
        
        reducer = UMAP(
            metric               = metric, 
            n_neighbors          = self.config.n_neighbors_umap, 
            n_components         = self.config.n_components, 
            n_epochs             = self.config.n_epochs, 
            learning_rate        = self.config.learning_rate, 
            init                 = self.config.init, 
            min_dist             = self.config.min_dist, 
            spread               = self.config.spread, 
            low_memory           = self.config.low_memory, 
            set_op_mix_ratio     = self.config.set_op_mix_ratio, 
            local_connectivity   = self.config.local_connectivity, 
            repulsion_strength   = self.config.repulsion_strength, 
            negative_sample_rate = self.config.negative_sample_rate, 
            transform_queue_size = self.config.transform_queue_size, 
            metric_kwds          = self.config.metric_kwds, 
            target_n_neighbors   = self.config.target_n_neighbors, 
            target_metric        = self.config.target_metric_umap, 
            target_metric_kwds   = self.config.target_metric_kwds, 
            target_weight        = self.config.target_weight, 
            transform_seed       = self.config.transform_seed, 
            random_state         = self.config.random_state
            )
        
        embedding = reducer.fit_transform(input_data)
        
        self.add_data("informed_reducer",   reducer)
        self.add_data("informed_embedding", pd.DataFrame(embedding))
        
        redshift = self.get_data("training_redshift")
        
        knn = NearestNeighbors(
            n_neighbors = self.config.n_neighbors_knn,
            metric = self.config.metric_knn
            )
        knn.fit(embedding)
        
        informed_knn = dict(
            model = knn,
            redshift = redshift.values
        )
        
        self.add_data("informed_knn", informed_knn)
        
    
    def UMAP_estimator(self):
        
        reducer = self.get_data("informed_reducer")
        
        photometry = self.get_data("estimation_photometry")
        
        if reducer.metric == manhattan_weighted_linear:
            
            phot_error = self.get_data("estimation_phot_error")
            input_data = pd.concat([photometry, phot_error], axis = 1)    
        
        else:
            input_data = photometry
        
        estimated_embedding = reducer.transform(input_data)
        
        informed_knn       = self.get_data("informed_knn")
        
        _, indices = informed_knn["model"].kneighbors(estimated_embedding)
        photozs = np.median(informed_knn["redshift"][indices], axis = 1)
        
        self.add_data("estimated_embedding", pd.DataFrame(estimated_embedding))
        self.add_data("estimated_photozs",   pd.DataFrame({"z_phot": photozs}))
        
        