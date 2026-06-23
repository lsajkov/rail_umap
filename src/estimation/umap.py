from rail.core.stage import RailStage
from rail.core.data import PqHandle
from ceci.config import StageParameter
from umap import UMAP

class UMAPEstimator(RailStage):
    
    name = 'UMAP'
    inputs = [("catalog_with_uncertainties", PqHandle)]
    outputs = [("outputs", PqHandle)]
    
    config_options = RailStage.config_options.copy()
    
    config_options.update(dict(
        inputType = StageParameter()
        n_components          = len(magnitude_columns)
n_neighbors           = 80
min_dist              = 0.0
    ))
    
    def run(self):
        
        catalog_with_uncertainties = self.get_data("catalog_with_uncertainties")
        