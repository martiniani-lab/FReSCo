import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import os, re, sys
from scipy.special import gamma, gammaln
import cmath
import daiquiri, logging
import inspect
import dill as pickle
import hickle as hkl
import pickletools
import gzip
import contextlib
import finufft
from tqdm import tqdm

def export_structure(r,L,Nk = 1001,radii=None,tol=1e-9):
    ndim = len(L)
    LL = np.array(np.reshape(L,(1,-1)))
    rr = np.array(r).reshape((-1, ndim))
    #rr -= np.round(rr/LL)*LL
    rr /= LL
    if np.amax(rr)>1:
        rr -= 0.5
    rr *= 2*np.pi
    N = rr.shape[0]
    if radii is None:
        c = np.ones(N)+0j
    else:
        if ndim == 2:
            c = np.pi*radii**2+0j
        elif ndim == 3:
            c = 4/3*np.pi*radii**3+0j
        else:
            print('Invalid dimension')
        c *= N/np.sum(c)
    center = int(Nk/2)
    if ndim == 2:
        Sk = finufft.nufft2d1(rr[:,0],rr[:,1],c,eps=tol,n_modes = (Nk,Nk))
        Sk = (np.absolute(Sk)**2)/N
        gr = np.fft.ifft2(np.fft.ifftshift(Sk)-1.,(Nk,Nk))
        gr = np.real(np.fft.fftshift(gr))*Nk*Nk*np.prod(L)/N
    elif ndim == 3:
        rho = finufft.nufft3d1(rr[:,0],rr[:,1],rr[:,2],c,eps=tol,n_modes = (Nk,Nk,Nk))
        Sk = (np.absolute(rho)**2)/N
        gr = np.fft.ifftn(np.fft.ifftshift(Sk)-1.,(Nk,Nk,Nk))
        gr = np.real(np.fft.fftshift(gr))*(Nk**ndim)*np.prod(L)/N
    else:
        print('Invalid dimension')
        return
    q,S = radial_profile(Sk,center = [center]*ndim)
    r,g = radial_profile(gr, center = [center]*ndim)
    return q,S,Sk,r/Nk,g, gr

def export_ewald(r,L,k0range,kvecs,radii=None,tol=1e-9):
    ndim = len(L)
    LL = np.array(np.reshape(L,(1,-1)))
    rr = np.array(r).reshape((-1, ndim))
    #rr -= np.round(rr/LL)*LL
    rr /= LL
    if np.amax(rr)>1:
        rr -= 0.5
    rr *= 2*np.pi
    print(np.amax(rr))
    print(np.amin(rr))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect('equal')
    ax.scatter(rr[:,0],rr[:,1])
    plt.show()
    plt.close()

    N = rr.shape[0]
    kvecs /= np.linalg.norm(kvecs,axis=-1).reshape(kvecs.shape[0],kvecs.shape[1],1)
    kmags = np.linalg.norm(k0range,axis=-1)
    # shape is (kmag, k0theta, theta, ndim)
    q = kmags.reshape(-1,kvecs.shape[0],1,1)*kvecs.reshape(1,kvecs.shape[0],kvecs.shape[1],ndim)-k0range.reshape(-1,kvecs.shape[0],1,ndim)
    qq = q.reshape(-1,ndim)
    if radii is None:
        c = np.ones(N)+0j
    else:
        if ndim == 2:
            c = np.pi*radii**2+0j
        elif ndim == 3:
            c = 4/3*np.pi*radii**3+0j
        else:
            print('Invalid dimension')
        c *= N/np.sum(c)
    if ndim == 2:
        ewald = finufft.nufft2d3(rr[:,0],rr[:,1],c,qq[:,0], qq[:,1], eps=tol)
    elif ndim == 3:
        ewald = finufft.nufft3d3(rr[:,0],rr[:,1],rr[:,2],c,qq[:,0],qq[:,1],qq[:,2],eps=tol)
    else:
        print('Invalid dimension')
        return
    return q,ewald.reshape(q.shape[0:3])

def export_singleTM(r,L,k0range,kvecs,radii=None,tol=1e-9):
    ndim = len(L)
    LL = np.array(np.reshape(L,(1,-1)))
    rr = np.array(r).reshape((-1, ndim))
    #rr -= np.round(rr/LL)*LL
    rr -= LL/2
    rr *= 2*np.pi/LL
    print(np.amax(rr))
    print(np.amin(rr))
    N = rr.shape[0]
    kvecs /= np.linalg.norm(kvecs,axis=-1).reshape(kvecs.shape[0],kvecs.shape[1],1)
    kmags = np.linalg.norm(k0range,axis=-1)
    theta = beamthetas[angle]
    cost,sint = onp.cos(-theta),onp.sin(-theta)
    u_ = onp.array([sint, cost])
    rot = onp.array([[cost,-sint],[sint,cost]])
    rrot = onp.matmul(rot,points.T).T #(rparallel, rperp)
    a = 2*rrot[:,1]/(w*w*k0)
    E0j = onp.exp(1j*rrot[:,0]*k0-(rrot[:,1]**2/(w*w*(1+1j*a))))/onp.sqrt(1+1j*a)

    # shape is (kmag, k0theta, theta, ndim)
    q = kmags.reshape(-1,kvecs.shape[0],1,1)*kvecs.reshape(1,kvecs.shape[0],kvecs.shape[1],ndim)-k0range.reshape(-1,kvecs.shape[0],1,ndim)
    qq = q.reshape(-1,ndim)
    if radii is None:
        c = np.ones(N)+0j
    else:
        if ndim == 2:
            c = np.pi*radii**2+0j
        elif ndim == 3:
            c = 4/3*np.pi*radii**3+0j
        else:
            print('Invalid dimension')
        c *= N/np.sum(c)
    if ndim == 2:
        ewald = finufft.nufft2d3(rr[:,0],rr[:,1],c,qq[:,0], qq[:,1], eps=tol)
    elif ndim == 3:
        ewald = finufft.nufft3d3(rr[:,0],rr[:,1],rr[:,2],c,qq[:,0],qq[:,1],qq[:,2],eps=tol)
    else:
        print('Invalid dimension')
        return
    return q,ewald.reshape(q.shape[0:3])
def radial_profile(data, center=None):
    ndim = len(data.shape)
    R = int(data.shape[0]/2)
    if center is None:
        center = np.array(data.shape)/2
    for i in range(ndim):
        center = np.expand_dims(center, axis=-1)
    idx = np.indices((data.shape))
    r = np.sqrt(np.sum((idx - center)**2,axis=0))
    r = r.astype(int)
    r = np.where(r <= R, r, R+1)
    tbin = np.bincount(r.ravel(),data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin/nr
    return np.unique(r).ravel()[:-1], radialprofile[:-1]

def find_k(K, L):
    '''Returns the list of all k vectors for a 2D square box of length L such that |k| < K'''
    k = (0, 0)
    for nx in range(int(K * L[0] / (2 * np.pi) + 1)):
        for ny in range(int(K * L[1] / (2 * np.pi) + 1)):
            if 4 * np.pi**2 * (nx**2 / L[0]**2 + ny**2 / L[1]**2) < K**2:
                k = np.vstack((k, (2 * np.pi * nx, 2 * np.pi * ny), (-2 * np.pi * nx, 2 * np.pi * ny),
                               (2 * np.pi * nx, -2 * np.pi * ny), (-2 * np.pi * nx, -2 * np.pi * ny)))
    k = np.unique(k, axis=0)
    k = np.delete(k, int(k.shape[0] / 2), axis=0)
    return k

def trymakedir(path):
    """this function deals with common race conditions"""
    while True:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                # time.sleep might help here
                pass
        else:
            break

def get_workdir(path=None):
    if path is None:
        workdir = os.getcwd()
    else:
        workdir = os.path.abspath(path)
    return workdir


def get_digits(string, use_float=True):
    if use_float:
        sl = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", string)]
    else:
        sl = [int(s) for s in re.findall(r"[-\d]+", string)]
    return sl


def volume_nball(radius, n):
    volume = np.power(np.pi, n / 2) * np.power(radius, n) / gamma(n / 2 + 1)
    return volume


def log_volume_nball(radius, n):
    log_volume = n / 2.0 * np.log(np.pi) + n * np.log(radius) - gammaln(n / 2 + 1)


def volume_nball_avg(radii, n):
    volume = np.power(np.pi, n / 2) * np.mean(np.power(radii, n)) / gamma(n / 2 + 1)
    return volume


def surface_nball(radius, n):
    return 2*np.pi*volume_nball(radius, n-1)


def log_surface_nball(radius, n):
    log_surface = np.log(2*np.pi) + log_volume_nball(radius, n-1)
    return log_surface


def is_power2(num):
    """states if a number is a power of two"""
    return (num != 0) and ((num & (num - 1)) == 0)


def closest_pow2(x):
    possible_results = np.floor(np.log2(x)), np.ceil(np.log2(x))
    return int(2 ** min(possible_results, key=lambda z: abs(x - 2 ** z)))


def log_factorial(x):
    return gammaln(x + 1)


def log_binomial(n, m):
    return log_factorial(n) - log_factorial(m) - log_factorial(n - m)


def get_array_range(x, val_min, val_max, ref=None):
    xcopy = np.array(x)
    if ref is None:
        ref = xcopy
    else:
        assert x.size == ref.size
    idxs = ref > val_min
    idxs2 = ref[idxs] < val_max
    return (xcopy[idxs])[idxs2]


def finite_xy(x, y):
    assert x.size == y.size
    idxs = np.isfinite(x)
    idxs2 = np.isfinite(y[idxs])
    return (x[idxs])[idxs2], (y[idxs])[idxs2]


def float_less_or_equal(x, y):
    return x < y or is_close(x, y)


def float_more_or_equal(x, y):
    return x > y or is_close(x, y)

def shuffle(x):
    xnew = np.array(x)
    np.random.shuffle(xnew)
    return xnew

def slice_narray(lattice_, lattice_boxv_, fracl=1):
    if fracl < 1:
        sliced_lattice_boxv = np.array(lattice_boxv_*fracl, dtype='int')
        sliced_lattice = np.array(lattice_).reshape(lattice_boxv_[::-1])
        sliced_lattice = sliced_lattice[tuple([slice(0, l, None) for l in sliced_lattice_boxv[::-1]])]
        return sliced_lattice.ravel(), sliced_lattice_boxv
    else:
        return lattice_, lattice_boxv_

def is_close(a, b, rel_tol=1e-9, abs_tol=0.0, method='weak'):
    """
    code imported from math.isclose python 3.5
    """
    if method not in ("asymmetric", "strong", "weak", "average"):
        raise ValueError('method must be one of: "asymmetric",'
                         ' "strong", "weak", "average"')

    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError('error tolerances must be non-negative')

    if a == b:  # short-circuit exact equality
        return True
    # use cmath so it will work with complex or float
    if cmath.isinf(a) or cmath.isinf(b):
        # This includes the case of two infinities of opposite sign, or
        # one infinity and one finite number. Two infinities of opposite sign
        # would otherwise have an infinite relative tolerance.
        return False
    diff = abs(b - a)
    if method == "asymmetric":
        return (diff <= abs(rel_tol * b)) or (diff <= abs_tol)
    elif method == "strong":
        return (((diff <= abs(rel_tol * b)) and
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "weak":
        return (((diff <= abs(rel_tol * b)) or
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "average":
        return ((diff <= abs(rel_tol * (a + b) / 2) or
                 (diff <= abs_tol)))
    else:
        raise ValueError('method must be one of:'
                         ' "asymmetric", "strong", "weak", "average"')


def is_close_array(a, b, rel_tol=1e-9, abs_tol=0.0):
    a, b = np.array(a), np.array(b)
    assert a.size == b.size, "learnscapes isCloseArray, arrays size mismatch"
    return np.allclose(a, b, rtol=rel_tol, atol=abs_tol, equal_nan=False)


def is_perfect_square(a):
    return is_close(np.sqrt(a) ** 2, a)


def int_to_uint8(lattice):
    x = np.array(lattice)
    x[x < 0] = 0
    return x.astype('uint8')


def append(arr, values):
    """
    this function should give a qualitatively similar behaviour to python append
    append values to array. If an array/list/tuple stack it, else append a scalar
    """
    arr = np.asarray(arr)
    if isinstance(values, (list, tuple, np.ndarray)):
        values = np.asarray(values)
        if arr.size == 0:
            arr = np.empty(np.append(0, values.shape))
        arr = np.append(arr, [values], axis=0)
    else:
        arr = np.append(arr, values)
    return arr


class Logger(object):
    """
    https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback
    """

    def __init__(self, logger=None, fname=str(inspect.currentframe().f_code.co_name),
                 level=logging.DEBUG, **kwargs):
        if logger is None:
            daiquiri.setup(level=level, **kwargs)
            self.logger = daiquiri.getLogger(fname)
        else:
            self.logger = logger

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)


def sizeof_format(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def old_pickle_dump(data, fname, protocol=-1):
    pickled_data = pickletools.optimize(pickle.dumps(data, protocol=protocol))
    with open(fname, 'wb') as f:
        f.write(pickled_data)

def pickle_dump(data, fname, protocol=-1, compression='gzip'):
    if compression == 'gzip':
        pickled_data = pickletools.optimize(pickle.dumps(data, protocol=protocol))
        with gzip.open(fname, 'wb') as f:
            f.write(pickled_data)
    else:
        raise NotImplementedError

def old_pickle_load(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_load(fname):
    try:
        with gzip.open(fname, 'rb') as f:
            data = pickle.load(f)
            return data
    except:
        return old_pickle_load(fname)

def hickle_dump(data, fname, compression='gzip', shuffle=True, fletcher32=True):
    hkl.dump(data, fname, mode='w', compression=compression, shuffle=shuffle, fletcher32=fletcher32)

def hickle_load(fname):
    data = hkl.load(fname)
    return data

@contextlib.contextmanager
def silence():
    # open 2 fds
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
    # save the current file descriptors to a tuple
    save = os.dup(1), os.dup(2)
    # put /dev/null fds on 1 and 2
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    #run function
    try:
        yield
    except:
        # restore file descriptors so I can print the results
        os.dup2(save[0], 1)
        os.dup2(save[1], 2)
        # close the temporary fds
        os.close(null_fds[0])
        os.close(null_fds[1])
        raise
    # restore file descriptors so I can print the results
    os.dup2(save[0], 1)
    os.dup2(save[1], 2)
    # close the temporary fds
    os.close(null_fds[0])
    os.close(null_fds[1])

def progressbar(*args, **kwargs):
    return tqdm(*args, **kwargs)
