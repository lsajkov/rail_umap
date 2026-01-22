import numpy as np
from ceci.config import StageParameter as Param
from rail.estimation.estimator import CatInformer
from rail.core.data import QPHandle, TableHandle, Hdf5Handle
from rail.core.common_params import SHARED_PARAMS
import qp
import umap
import pandas as pd

def _computecolordata(df, ref_column_name, column_names, only_color):
    """
    EXACT same color feature construction pattern as the KNN example.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing band columns
    ref_column_name : str
        Reference band column name (typically a magnitude)
    column_names : list[str]
        Ordered list of band columns, e.g. ["g","r","i","z","y"]
    only_color : bool
        If True, do not include reference magnitude, only use colors.
        If False, include ref band magnitude as feature 'x'.

    Returns
    -------
    coldata : np.ndarray
        Feature matrix with shape (N, n_features)
    """
    newdict = {}
    if not only_color:
        newdict["x"] = df[ref_column_name]
    nbands = len(column_names) - 1
    for k in range(nbands):
        newdict[f"x{k}"] = df[column_names[k]] - df[column_names[k + 1]]
    newdf = pd.DataFrame(newdict)
    coldata = newdf.to_numpy()
    return coldata


def train_umap(
    X,
    n_neighbors=20,
    min_dist=0.1,
    metric="manhattan",
    n_components=3,
    init="spectral",
    verbose=True,
    n_epochs=100,
    random_state=None,
):
    """
    Train UMAP on a precomputed feature matrix X.

    Returns
    -------
    umap_model : umap.UMAP
        Trained UMAP model.
    embedding : np.ndarray
        Low-dimensional embedding.
    """
    umap_obj = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        init=init,
        verbose=verbose,
        n_epochs=n_epochs,
        random_state=random_state,
    )
    umap_model = umap_obj.fit(X)
    return umap_model


class UMAPTrainer(CatInformer):
    """Train a UMAP model from catalog-like inputs.
    """
    name = "UMAPTrainer"
    config_options = CatInformer.config_options.copy()
    config_options.update(
        # shared / io
        hdf5_groupname=SHARED_PARAMS,

        # feature construction (match KNN conventions)
        bands=SHARED_PARAMS,        # ordered list of band column names
        ref_band=SHARED_PARAMS,     # name of reference band column (must be in bands)
        only_colors=Param(bool, True, msg="If True, only use colors; else include ref_band mag as feature 'x'."),

        # nondetect handling (same idea as KNN)
        nondetect_val=SHARED_PARAMS,
        mag_limits=SHARED_PARAMS,   # dict-like: mag_limits[col] gives replacement value

        # UMAP hyperparameters
        n_neighbors=Param(int, 20, msg="UMAP n_neighbors"),
        min_dist=Param(float, 0.1, msg="UMAP min_dist"),
        metric=Param(str, "manhattan", msg="UMAP metric"),
        n_components=Param(int, 3, msg="UMAP embedding dimension"),
        init=Param(str, "spectral", msg="UMAP init"),
        verbose=Param(bool, True, msg="UMAP verbose"),
        n_epochs=Param(int, 100, msg="UMAP n_epochs"),
        random_state=Param(int, 0, msg="UMAP random_state"),
    )

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Basic sanity: bands must include ref_band
        if self.config.bands and self.config.ref_band:
            if self.config.ref_band not in self.config.bands:
                raise ValueError(
                    f"UMAPTrainer: ref_band='{self.config.ref_band}' must be in bands={self.config.bands}"
                )

    def run(self):
        # Load training data from input handle (optionally inside hdf5 group)
        if self.config.hdf5_groupname:
            training_data = self.get_data("input")[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data("input")

        # Build df with band columns in the specified order (same style as KNN)
        umapdf = pd.DataFrame(training_data, columns=self.config.bands)

        # Replace nondetects exactly like KNN example
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                umapdf.loc[np.isnan(umapdf[col]), col] = np.float32(self.config.mag_limits[col])
            else:
                umapdf.loc[np.isclose(umapdf[col], self.config.nondetect_val), col] = np.float32(
                    self.config.mag_limits[col]
                )

        # Compute color feature matrix using the SAME helper signature/pattern
        X = _computecolordata(
            umapdf,
            self.config.ref_band,
            self.config.bands,
            self.config.only_colors,
        )

        # Train UMAP
        umap_model = train_umap(
            X,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            n_components=self.config.n_components,
            init=self.config.init,
            verbose=self.config.verbose,
            n_epochs=self.config.n_epochs,
            random_state=self.config.random_state,
        )

        # Store model (and optionally embedding + feature metadata)
        self.model = dict(
            reducer=umap_model,         # handy for diagnostics; remove if you prefer
            bands=self.config.bands,
            ref_band=self.config.ref_band,
            only_colors=self.config.only_colors,
        )

        self.add_data("model", self.model)


    def umap_transform(self, umapdf, model):
        
        # Replace nondetects exactly like KNN example
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                umapdf.loc[np.isnan(umapdf[col]), col] = np.float32(self.config.mag_limits[col])
            else:
                umapdf.loc[np.isclose(umapdf[col], self.config.nondetect_val), col] = np.float32(
                    self.config.mag_limits[col]
                )
        # Compute color feature matrix using the SAME helper signature/pattern
        X = _computecolordata(
            umapdf,
            self.config.ref_band,
            self.config.bands,
            self.config.only_colors,
        )

        embedding = model.transform(X)

        return embedding





