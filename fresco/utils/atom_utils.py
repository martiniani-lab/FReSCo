import numpy as np
import itertools
from fresco.utils import volume_nball
from fresco.distances import Distance, get_distance, put_in_box
from numba import jit

def _get_particles_volume(radii, ndim):
    """returns volume of ndim-dimensional sphere"""
    volumes = volume_nball(radii, ndim)
    vtot = np.sum(volumes)
    return vtot


def resize_box(vol_frac, radii, boxv):
    """adjust the box size to meet the target vol fraction"""
    ndim = len(boxv)
    vol_part = _get_particles_volume(radii, ndim)
    vol_box = np.prod(boxv)
    phi = vol_part / vol_box  # instanteneous pack frac
    a = np.power(phi / vol_frac, 1. / ndim)
    boxv *= a
    # TEST
    vol_box = np.prod(boxv)
    phi = vol_part / vol_box
    assert (phi - vol_frac < 1e-4)
    # END TEST
    return boxv


def gen_rand_coords(natoms, boxv, centre=True):
    d = 0.5 if centre else 0
    ndim = len(boxv)
    coords = (np.random.rand(natoms * ndim) - d).reshape(-1, ndim) * boxv
    return coords.ravel()


@jit(nopython=True)
def dot(x):
    return np.dot(x, x)


@jit(nopython=True)
def _get_lattice_index(lattice_coordinates, lattice_boxv):
    ndim = len(lattice_boxv)
    x = lattice_coordinates
    if ndim == 1:
        return x[0]
    elif ndim == 2:
        return x[0] + lattice_boxv[0] * x[1]
    elif ndim == 3:
        return x[0] + lattice_boxv[0] * x[1] + lattice_boxv[0] * lattice_boxv[1] * x[2]
    else:
        raise NotImplementedError


@jit
def convert_to_lattice_pbc(coords_, boxv, lattice_boxv):
    coords = np.array(coords_)
    ndim = len(boxv)
    coords = put_in_box(coords, ndim, Distance.PERIODIC, boxv)
    coords = coords.reshape(-1, ndim)
    coords += (boxv / 2)  # shift coords in [0, boxv]
    bin_size = boxv / np.array(lattice_boxv)
    lattice_coords = np.floor(coords / bin_size).astype('int')
    lattice = np.zeros(np.prod(lattice_boxv))
    for x in lattice_coords:
        idx = _get_lattice_index(x, lattice_boxv)
        lattice[idx] += 1
    assert len(coords) == np.sum(lattice)
    return lattice


@jit
def convert_to_lattice_cartesian(coords_, boxv, lattice_boxv, xmin=None):
    coords = np.array(coords_, dtype='d')
    ndim = len(boxv)
    coords = coords.reshape(-1, ndim)
    xmin = np.zeros(ndim) if xmin is None else xmin
    coords -= xmin  # shift coords in [0, boxv]
    bin_size = boxv / np.array(lattice_boxv)
    lattice_coords = np.floor(coords / bin_size).astype('int')
    lattice = np.zeros(np.prod(lattice_boxv))
    for i, x in enumerate(lattice_coords):
        idx = _get_lattice_index(x, lattice_boxv)
        lattice[idx] += 1
    assert len(coords) == np.sum(lattice)
    return lattice


@jit
def _get_lattice_bin_centers(boxv, lattice_boxv):
    # https://stackoverflow.com/questions/4709510/itertools-product-speed-up
    ndim = len(lattice_boxv)
    bin_size = boxv / np.array(lattice_boxv)
    lattice_indices = np.rollaxis(np.indices(lattice_boxv), 0, ndim + 1)
    lattice_centers = (lattice_indices + 0.5) * bin_size
    return lattice_centers


def _get_pixellated_image_spheres_slow(coords, radii, boxv, lattice_boxv, distance=Distance.PERIODIC):
    # put in box and shift is done in the higher level function
    ndim = len(lattice_boxv)
    coords = coords.reshape(-1, ndim)
    boxv = np.asarray(boxv, dtype=np.float)
    im_array = np.zeros(lattice_boxv)
    lattice_centers = _get_lattice_bin_centers(boxv, lattice_boxv)
    for idx, xc in enumerate(coords):
        r2 = radii[idx] * radii[idx]
        for t in itertools.product(*[range(n) for n in lattice_boxv]):
            xs = lattice_centers[t]
            d2 = dot(get_distance(xc, xs, ndim, distance, boxv))
            im_array[t[::-1]] += 1. if d2 < r2 else 0.
    return im_array


def _get_pixellated_image_spheres(coords, radii, boxv, lattice_boxv, distance=Distance.PERIODIC):
    # put in box and shift is done in the higher level function
    ndim = len(lattice_boxv)
    coords = coords.reshape(-1, ndim)
    boxv = np.asarray(boxv, dtype=np.float)
    bin_size = boxv / np.array(lattice_boxv)
    im_array = np.zeros(shape=lattice_boxv[::-1])
    lattice_centers = _get_lattice_bin_centers(boxv, lattice_boxv)
    lattice_coords = np.floor(coords / bin_size).astype('int')
    for idx, (xc, lc) in enumerate(zip(coords, lattice_coords)):
        r2 = radii[idx] * radii[idx]
        lr = np.ceil(radii[idx] / bin_size).astype('int') + 1
        if distance == Distance.PERIODIC:
            window = [range(p - lr[i], p + lr[i] + 1) for i, p in enumerate(lc)]
        elif distance == Distance.CARTESIAN:
            window = [range(max(0, p - lr[i]), min(lattice_boxv[i], p + lr[i] + 1)) for i, p in enumerate(lc)]
        else:
            raise NotImplementedError
        for t in itertools.product(*window):
            t = tuple(np.mod(t, lattice_boxv))
            xs = lattice_centers[t]
            d2 = dot(get_distance(xc, xs, ndim, distance, boxv))
            im_array[t[::-1]] += 1. if d2 < r2 else 0.
    return im_array


def convert_to_lattice_pix_spheres_pbc(coords_, radii, boxv, lattice_boxv):
    coords = np.array(coords_)
    ndim = len(boxv)
    boxv = np.asarray(boxv, dtype=np.float)
    coords = put_in_box(coords, ndim, Distance.PERIODIC, boxv)
    coords = coords.reshape(-1, ndim)
    coords += (boxv / 2)  # shift coords in [0, boxv]
    im_array = _get_pixellated_image_spheres(coords, radii, boxv, lattice_boxv, distance=Distance.PERIODIC)
    lattice = np.ravel(im_array)
    return lattice


def convert_to_lattice_pix_spheres_cartesian(coords_, radii, boxv, lattice_boxv, xmin=None):
    coords = np.array(coords_)
    ndim = len(boxv)
    boxv = np.asarray(boxv, dtype=np.float)
    coords = coords.reshape(-1, ndim)
    xmin = np.zeros(ndim) if xmin is None else xmin
    coords -= xmin  # shift coords in [0, boxv]
    im_array = _get_pixellated_image_spheres(coords, radii, boxv, lattice_boxv, distance=Distance.CARTESIAN)
    lattice = np.ravel(im_array)
    return lattice


def convert_to_lattice(coords, boxv, lattice_boxv, distance, nopix=False, radii=None, xmin=None):
    if distance == Distance.CARTESIAN:
        if radii is None or nopix:
            return convert_to_lattice_cartesian(coords, boxv, lattice_boxv, xmin=xmin)
        else:
            return convert_to_lattice_pix_spheres_cartesian(coords, radii, boxv, lattice_boxv, xmin=xmin)
    elif distance == Distance.PERIODIC:
        if radii is None or nopix:
            return convert_to_lattice_pbc(coords, boxv, lattice_boxv)
        else:
            return convert_to_lattice_pix_spheres_pbc(coords, radii, boxv, lattice_boxv)
    else:
        raise NotImplementedError
