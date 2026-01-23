"""
quick implementation of k nearest neighbor estimator
First pass will ignore photometric errors and just do
things in terms of magnitudes, we will expand in a
future update
"""

import numpy as np
import copy

from ceci.config import StageParameter as Param
from rail.core.data import ModelHandle, TableHandle, QPHandle
from rail.estimation.estimator import CatEstimator, CatInformer

from rail.evaluation.metrics.cdeloss import CDELoss
from rail.core.common_params import SHARED_PARAMS

import pandas as pd
import qp
import umap


TEENY = 1.e-15

def _computereduceddata(df, ref_column_name, column_names, only_color, reducer):
    newdict = {}
    if only_color:
        newdict['x'] = df[ref_column_name]
    nbands = len(column_names) - 1
    for k in range(nbands):
        newdict[f'x{k}'] = df[column_names[k]] - df[column_names[k + 1]]
    newdf = pd.DataFrame(newdict)
    coldata = newdf.to_numpy()
    embeddata = reducer.transform(coldata)
    return embeddata


def _makepdf(dists, ids, szs, sigma):
    sigmas = np.full_like(dists, sigma)
    weights = 1. / dists
    weights /= weights.sum(axis=1, keepdims=True)
    means = szs[ids]
    pdfs = qp.Ensemble(qp.mixmod, data=dict(means=means, stds=sigmas, weights=weights))
    return pdfs


class UmapKnnInformer(CatInformer):
    """Train a KNN-based estimator
    """
    name = 'UmapKnnInformer'
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          hdf5_groupname=SHARED_PARAMS,
                          trainfrac=Param(float, 0.75,
                                          msg="fraction of training data used to make tree, rest used to set best sigma"),
                          seed=Param(int, 0, msg="Random number seed for NN training"),
                          sigma_grid_min=Param(float, 0.01, msg="minimum value of sigma for grid check"),
                          sigma_grid_max=Param(float, 0.075, msg="maximum value of sigma for grid check"),
                          ngrid_sigma=Param(int, 10, msg="number of grid points in sigma check"),
                          leaf_size=Param(int, 15, msg="min leaf size for KDTree"),
                          nneigh_min=Param(int, 3, msg="int, min number of near neighbors to use for PDF fit"),
                          nneigh_max=Param(int, 7, msg="int, max number of near neighbors to use ofr PDF fit"))
    inputs = [("model", ModelHandle),
              ("input", TableHandle)]
    outputs = [("output_model", ModelHandle)]

    def __init__(self, args, **kwargs):
        """ Constructor
        Do CatInformer specific initialization, then check on bands """
        super().__init__(args, **kwargs)

        usecols = self.config.bands.copy()
        usecols.append(self.config.redshift_col)
        self.usecols = usecols
        self.zgrid = None

    def open_model(self, **kwargs):
        CatInformer.open_model(self, **kwargs)
        self.reducer = self.model['reducer']
        self.only_colors = self.model['only_colors']

    def run(self):
        """
        train a KDTree on a fraction of the training data
        """
        from sklearn.neighbors import KDTree
        self.open_model(**self.config)

        print(f"value of only colors: {self.only_colors}")
        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data('input')
        knndf = pd.DataFrame(training_data, columns=self.config.bands)
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                knndf.loc[np.isnan(knndf[col]), col] = np.float32(self.config.mag_limits[col])
            else:
                knndf.loc[np.isclose(knndf[col], self.config.nondetect_val), col] = np.float32(self.config.mag_limits[col])

        trainszs = np.array(training_data[self.config.redshift_col])
        reduceddata = _computereduceddata(knndf, self.config.ref_band, self.config.bands, self.only_colors,
                                                      self.reducer)
        nobs = reduceddata.shape[0]
        rng = np.random.default_rng(seed=self.config.seed)
        perm = rng.permutation(nobs)
        ntrain = round(nobs * self.config.trainfrac)
        xtrain_data = reduceddata[perm[:ntrain]]
        train_data = copy.deepcopy(xtrain_data)
        val_data = reduceddata[perm[ntrain:]]
        xtrain_sz = trainszs[perm[:ntrain]].copy()
        train_sz = np.array(copy.deepcopy(xtrain_sz))
        val_sz = np.array(trainszs[perm[ntrain:]])
        print(f"split into {len(train_sz)} training and {len(val_sz)} validation samples")
        tmpmodel = KDTree(train_data, leaf_size=self.config.leaf_size)
        # Find best sigma and n_neigh by minimizing CDE Loss
        bestloss = 1e20
        bestsig = self.config.sigma_grid_min
        bestnn = self.config.nneigh_min
        siggrid = np.linspace(self.config.sigma_grid_min, self.config.sigma_grid_max, self.config.ngrid_sigma)
        print("finding best fit sigma and NNeigh...")
        for sig in siggrid:
            for nn in range(self.config.nneigh_min, self.config.nneigh_max + 1):
                dists, idxs = tmpmodel.query(val_data, k=nn)
                # add a small small number to guard against NaN when obj of same color exists in spec file
                dists += TEENY
                ens = _makepdf(dists, idxs, train_sz, sig)
                cdelossobj = CDELoss(ens, self.zgrid, val_sz)
                cdeloss = cdelossobj.evaluate().statistic
                if cdeloss < bestloss:
                    bestsig = sig
                    bestnn = nn
                    bestloss = cdeloss
        numneigh = bestnn
        sigma = bestsig
        print(f"\n\n\nbest fit values are sigma={sigma} and numneigh={numneigh}\n\n\n")
        # remake tree with full dataset!
        kdtree = KDTree(reduceddata, leaf_size=self.config.leaf_size)
        self.model = dict(kdtree=kdtree,
                          bestsig=sigma,
                          nneigh=numneigh,
                          truezs=trainszs,
                          only_colors=self.only_colors,
                          reducer=self.reducer)
        self.add_data('output_model', self.model)


class UmapKnnEstimator(CatEstimator):
    """KNN-based estimator
    """
    name = 'UmapKnnEstimator'
    config_options = CatEstimator.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS)
    inputs = [("model", ModelHandle),
              ("input", TableHandle)]
    outputs = [("output", QPHandle)]

    def __init__(self, args, **kwargs):
        """ Constructor:
        Do Estimator specific initialization """
        self.sigma = None
        self.numneigh = None
        self.model = None
        self.trainszs = None
        self.zgrid = None
        self.only_colors = None
        super().__init__(args, **kwargs)
        usecols = self.config.bands.copy()
        usecols.append(self.config.redshift_col)
        self.usecols = usecols

    def open_model(self, **kwargs):
        CatEstimator.open_model(self, **kwargs)
        if self.model is None:   # pragma: no cover
            return
        self.sigma = self.model['bestsig']
        self.numneigh = self.model['nneigh']
        self.kdtree = self.model['kdtree']
        self.trainszs = self.model['truezs']
        self.only_colors = self.model['only_colors']
        self.reducer = self.model['reducer']
        print(f"value of only colors: {self.only_colors}")
        
    def _process_chunk(self, start, end, data, first):
        """
        calculate and return PDFs for each galaxy using the trained flow
        """
        print(f"Process {self.rank} estimating PZ PDF for rows {start:,} - {end:,}")
        knn_df = pd.DataFrame(data, columns=self.config.bands)
        self.zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins)

        # replace nondetects
        # will fancy this up later with a flow to sample from truth
        for col in self.config.bands:
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                knn_df.loc[np.isnan(knn_df[col]), col] = np.float32(self.config.mag_limits[col])
            else:
                knn_df.loc[np.isclose(knn_df[col], self.config.nondetect_val), col] = np.float32(self.config.mag_limits[col])

        testcolordata = _computereduceddata(knn_df, self.config.ref_band, self.config.bands, self.only_colors,
                                            self.reducer)
        dists, idxs = self.kdtree.query(testcolordata, k=self.numneigh)
        dists += TEENY
        test_ens = _makepdf(dists, idxs, self.trainszs, self.sigma)

        zmode = test_ens.mode(grid=self.zgrid)
        test_ens.set_ancil(dict(zmode=zmode))
        self._do_chunk_output(test_ens, start, end, first, data=data)
