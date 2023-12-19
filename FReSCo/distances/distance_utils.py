import numpy as np
from FReSCo.distances import Distance, get_distance
from FReSCo.utils import omp_get_thread_count
from numba import jit

def get_ncellsx_scale(radii, boxv, omp_threads=None):
    if omp_threads is None:
        omp_threads = omp_get_thread_count()
    ndim = len(boxv)
    ncellsx_max = max(omp_threads, int(np.power(radii.size, 1. / ndim)))
    rcut = np.amax(radii) * 2
    ncellsx = 1 * boxv[0] / rcut
    if ncellsx <= ncellsx_max:
        ncellsx_scale = 1 if ncellsx >= omp_threads else np.ceil(omp_threads / ncellsx)
    else:
        ncellsx_scale = ncellsx_max / ncellsx
    print("ncellsx: {}, ncellsx_scale: {}".format(ncellsx, ncellsx_scale))
    return ncellsx_scale

def get_rcut_scaled(radii, boxv):
    ndim = len(boxv)
    ncellsx_max = int(np.power(radii.size, 1. / ndim))
    rcut = np.amax(radii) * 2
    ncellsx = 1 * boxv[0] / rcut
    ncellsx_scale = 1 if ncellsx < ncellsx_max else ncellsx_max / ncellsx
    return boxv[0] / (ncellsx_scale * ncellsx)

@jit
def count_contacts(coords, radii, boxv):
    ndim = len(boxv)
    z = np.zeros(coords.size // ndim)
    coords = coords.reshape(-1, ndim)
    for i in xrange(len(coords)):
        for j in xrange(i + 1, len(coords)):
            d = get_distance(coords[i], coords[j], ndim, Distance.PERIODIC, box=boxv)
            if np.linalg.norm(d) < radii[i] + radii[j]:
                z[i] += 1
                z[j] += 1
    return z
