from functools import lru_cache

import time
import json
import numpy as np
import scipy.sparse as sps
from numba import njit
from sklearn.neighbors import NearestNeighbors
from skimage.color import rgb2lab
from skimage.util import img_as_float

njit = njit(cache=True)


@njit
def _find(forest, i):
    if forest[i] != i:
        forest[i] = _find(forest, forest[i])
    return forest[i]


@njit
def _union(forest, a, b):
    forest[_find(forest, b)] = forest[_find(forest, a)]


@njit
def _tomato_pre(density, neighbors):
    n = len(density)
    forest = np.arange(n)

    ind = density.argsort()[::-1]
    order = ind.argsort()

    for i in ind:
        for j in neighbors[i]:
            if order[j] > order[i]:
                continue
            forest[i] = j

    return forest, order, ind


@njit
def _tomato(density, neighbors, tau, forest, order, ind):
    forest = forest.copy()

    for i in ind:
        if neighbors.shape[1] == 1 or tau == 0:
            continue
        for j in neighbors[i]:
            if order[j] > order[i]:
                continue
            ri, rj = _find(forest, i), _find(forest, j)
            if ri != rj and min(density[ri], density[rj]) < density[i] + tau:
                if order[ri] < order[rj]:
                    _union(forest, ri, rj)
                else:
                    _union(forest, rj, ri)

    for i in range(len(density)):
        _find(forest, i)
    return forest


def normalize_clusters(y):
    _, index, inverse = np.unique(y, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(index))
    return order[inverse]


def tomato(
    points,
    *,
    k,
    tau=None,
    n_clusters=None,
    relative_tau: bool = True,
    keep_cluster_labels: bool = False,
    rmsd_path='',
    density_path='',
    neighbors_path='',
):
    """ToMATo clustering

    Parameters
    ----------

    points : np.ndarray
        Array of shape (n, dim)
    k : int
        Number of nearest neighbors to build the graph with
    tau : float or None
        Prominence threshold. Must not be specified if `n_clusters` is given.
    relative_tau : bool
        If `relative_tau` is set to `True`, `tau` will be multiplied by the standard deviation of the densities, making easier to have a unique value of `tau` for multiple datasets.
    n_clusters : int or None
        Target number of clusters. Must not be specified if `tau` is given.
    keep_cluster_labels : bool
        If False, converts the labels to make them contiguous and start from 0.
    rmsd_path : str
        If '', computes the euclidean distance to the nearest neighbors using sklearn.
        Else, recover the rmsd matrix as distance from the given path.
    density_path : str
        If '', and rmsd_path is also '' : computes the euclidean distance to the nearest neighbors using sklearn.
        Elif '', and rmsd_path is not '' : computes the density matrix out of the rmsd one.
        Else, recover the density matrix from the given path.
    neighbors_path : str
        Same as density_path.

    Returns
    -------

    clusters : np.ndarray
        Array of shape (n,) containing the cluster indexes.
    tau : float
        Prominence threshold. Only present if `n_clusters` was given.

    """

    assert [tau, n_clusters].count(
        None
    ) == 1, "You cannot give both `tau` and `n_clusters`"
    assert n_clusters is None or n_clusters > 0

    if rmsd_path=='':
        distances, neighbors = NearestNeighbors(n_neighbors=k).fit(points).kneighbors()
    elif density_path=='':
        extension = rmsd_path.split('.')[-1]
        if extension=='npz':
            rmsd = sps.load_npz(rmsd_path)
            distances, keys, neighbors = rmsd.data, rmsd.row, rmsd.col
        elif extension=='json':
            with open(rmsd_path) as json_file:
                rmsd_file = json.load(json_file)

            keys, distances, neighbors = rmsd_file['rows'], rmsd_file['values'], rmsd_file['cols']
            keys, distances, neighbors = np.array(keys), np.array(distances), np.array(neighbors)

        nb_neighbors = len(keys[keys==0])
        neighbors = neighbors.reshape(-1,nb_neighbors)
        distances = distances.reshape(-1,nb_neighbors)

        neighbors = np.concatenate([np.array([elt for elt in range(distances.shape[0])]).reshape(-1,1), neighbors], axis=1)
        distances = np.concatenate([np.array([0 for _ in range(distances.shape[0])]).reshape(-1,1), distances], axis=1)

        if k is not None:
            neighbors, distances = neighbors[:,:k], distances[:,:k]

    if density_path=='':
        density = ((distances ** 2).mean(axis=-1) + 1e-10) ** -0.5
    else:
        density = np.load(density_path)
        neighbors = np.load(neighbors_path)

    pre = _tomato_pre(density, neighbors)

    if tau is not None:
        if relative_tau:
            tau *= density.std()
        ans = _tomato(density, neighbors, tau, *pre)
    else:

        @lru_cache(1)
        def aux1(tau):
            return _tomato(density, neighbors, np.float32(tau), *pre)

        def aux2(tau):
            return len(np.unique(aux1(tau)))

        if aux2(0) < n_clusters:
            # error
            tau = -1
            ans = aux1(0)
        else:
            a = 0
            b = density.max() - density.min() + 1

            if aux2(b) > n_clusters:
                # error
                tau = -1
                ans = aux1(b)
            else:
                # binary search
                while aux2((a + b) / 2) != n_clusters:
                    print(a, b, aux2((a + b) / 2))
                    if aux2((a + b) / 2) > n_clusters:
                        a = (a + b) / 2
                    else:
                        b = (a + b) / 2

            tau = (a + b) / 2
            ans = aux1(tau)

    if not keep_cluster_labels:
        ans = normalize_clusters(ans)

    if n_clusters is None:
        return ans
    else:
        return ans, tau
