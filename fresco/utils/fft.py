import numpy as np
from scipy.signal import boxcar
from pyfftw.interfaces.numpy_fft import fft, fftshift, fftfreq, fftn, ifftn, ifftshift
from fresco.utils import DistanceValueHistogram, is_power2
import pyfftw
from pynfft.nfft import NFFT


def fft1d(y, d=1):
    """
    y: 1D array
        signal
    d: float
        sample spacing / timestep (inverse of the sampling rate). Defaults to 1.
    """
    y = np.asarray(y, dtype='d')
    sp = fftshift(fft(y, norm='ortho'))
    freq = fftshift(fftfreq(y.size, d=d))
    return freq, np.abs(sp)


def coords_to_grid(coords, d=1e-2):
    """
    coords: 1D array
            atomic coordinates
    d: float
        sample spacing / timestep (inverse of the sampling rate). Defaults to 1e-2.
    """
    grid = np.zeros(np.ceil(np.amax(coords) / d).astype('int') + 1)
    grid[np.floor(coords / d).astype('int')] = 1
    return grid


def coords_fft1d(coords, d=1e-2):
    """
    coords: 1D array
            atomic coordinates
    d: float
        sample spacing / timestep (inverse of the sampling rate). Defaults to 1e-2.
    """
    grid = coords_to_grid(coords, d=d)
    freq, ampl = fft1d(grid, d=d)
    return freq, ampl


def structure_factor_1d(x, dx=1, norm=None):
    norm = np.sum(x) if norm is None else norm
    sp = fftshift(fft(x))
    freq = fftshift(fftfreq(x.size, d=dx))
    # 1/\rho <\rho(k) \rho(-k)> (need to divide by N to normalise fft)
    sq = np.real(sp * np.conj(sp)) / norm
    return freq[freq > 0], sq[freq > 0]


def _get_freqs(shape, dx=1):
    ndim = len(shape)
    freq2 = 0
    for i in xrange(ndim):
        freq2_ = fftshift(fftfreq(shape[i], d=dx)) ** 2
        freq2 = np.add.outer(freq2, freq2_)
    freq = np.sqrt(freq2)
    return freq


def structure_factor_n(x, ndim, dx=1, norm=None):
    norm = np.sum(x) if norm is None else norm
    assert ndim == len(x.shape)
    sp = fftshift(fftn(x))
    sq = np.real(sp * np.conj(sp)) / norm
    freq = _get_freqs(x.shape, dx=dx)
    assert freq.size == sq.size
    return freq, sq  # returns the norm of the vector frequencies and sq


def structure_factor_nfft_n(x_, boxv, nbins=1024, interlace=False):
    assert is_power2(nbins), "nfft requires nbins to be a power of 2"
    assert (boxv[0] == boxv).all(), "boxv is not square"
    ndim = len(boxv)
    x = np.reshape(x_, (-1, ndim)) / boxv
    nodes = x.shape[0]
    bins = [nbins] * ndim  # use a power of 2
    dx = boxv[0] / nbins  # assumes that distance units are the same in x, y, z etc.
    # take the fft
    plan = NFFT(bins, nodes)
    plan.x = x  # assign coordinates
    plan.precompute()
    plan.f = np.ones((nodes, 1))  # weight points equally
    fhat = plan.adjoint()
    if interlace:
        qs = np.meshgrid(*([np.arange(-nbins // 2, nbins // 2)] * ndim))
        intmask = np.exp(-np.imag(1) * np.pi * np.sum(qs, axis=0) / nbins)
        plan.x = x + np.ones(ndim) / (2 * nbins)  # assumes that coordinates have been normalised to [0, 1)
        plan.precompute()
        fhatshift = plan.adjoint()
        fhat = (fhat + intmask * fhatshift) / 2.
    sq_ = fhat * np.conj(fhat)
    # compute gr, note numpy ifft assumes 1/n normalization by default. Also must divide by number density g(r) = 1/\rho IFT(S(q) - 1)
    gr_ = ifftn(ifftshift(sq_/nodes - 1.))
    autocor = np.real(fftshift(gr_)) * nbins**2 * np.prod(boxv) / nodes
    sq = np.real(sq_) / nodes
    freq = _get_freqs(bins, dx=dx)
    r = _get_distances(autocor.shape, dx=dx)
    return (freq, sq), (r, autocor)


def _angular_average(freq, sq):
    dvh = DistanceValueHistogram()
    for d, v in zip(freq.ravel(), sq.ravel()):
        dvh.update(d, v)
    return dvh


def angular_average(freq, sq, nbins=None, dmin=None, dmax=None, mincount=1, geomspace=False):
    nbins = min(5 * np.unique(freq).size, 10000) if nbins is None else nbins
    dmax = np.amax(freq) if dmax is None else dmax
    dmin = np.amin(freq) if dmin is None else dmin
    dvh = _angular_average(freq, sq)
    x, y = dvh.get_distance_values_hist(nbins, dmin=dmin, dmax=dmax, mincount=mincount, geomspace=geomspace)
    return x, y


def dynamic_structure_factor_1d(xs, dx=1, dt=1, norm=None):
    # expects vstacked configurations
    norm = np.sum(xs) if norm is None else norm
    nx, nt = len(xs[0]), len(xs)
    w = boxcar(nt)
    w = np.multiply.outer(w, np.ones(nx))
    assert w.shape == xs.shape
    sp = fftshift(fftn(xs) * w)
    dsq = np.real(sp * np.conj(sp)) / norm
    freqx = fftshift(fftfreq(nx, d=dx))
    freqt = fftshift(fftfreq(nt, d=dt))
    # sq = sq[:, freqx>0]
    # sq = sq[freqt>0, :]
    return freqt, freqx, dsq


def dynamic_structure_factor_n(xs, dx=1, dt=1, norm=None):
    # expects vstacked configurations
    norm = np.sum(xs) if norm is None else norm
    shapex, nt = xs[0].shape, len(xs)
    ndim = len(shapex)
    w = boxcar(nt)
    for i in xrange(ndim):
        w = np.multiply.outer(w, np.ones(shapex[i]))
    assert w.shape == xs.shape
    sp = fftshift(fftn(xs) * w)
    dsq = np.real(sp * np.conj(sp)) / norm
    freqx = _get_freqs(shapex, dx=dx)
    freqt = fftshift(fftfreq(nt, d=dt))
    return freqt, freqx, dsq  # you might want to angular average each elements of sq


def _dynamic_angular_average(freqt, freqx, dsq):
    dvh_list = []
    for sqx in dsq:
        dvh = DistanceValueHistogram()
        for d, v in zip(freqx.ravel(), sqx.ravel()):
            dvh.update(d, v)
        dvh_list.append(dvh)
    assert len(dvh_list) == len(freqt)
    return dvh_list


def dynamic_angular_average(freqt, freqx, dsq, nbins=None, dmax=None, mincount=1, geomspace=False):
    nbins = 5 * np.unique(freqx).size if nbins is None else nbins
    dmax = np.amax(freqx) if dmax is None else dmax
    xs, zs = [], []  # the frequencies is typically y
    dvh_list = _dynamic_angular_average(freqt, freqx, dsq)
    for dvh in dvh_list:
        x, z = dvh.get_distance_values_hist(nbins, dmax, mincount=mincount, geomspace=geomspace)
        xs.append(x)
        zs.append(z)
    assert len(xs) == len(freqt)
    return dvh, np.array(xs), freqt, np.array(zs)  # there should be one pair for each freqt


def _get_distances(shape, dx=1):
    shape = np.asarray(shape, dtype='int')
    d = np.empty(shape)
    c = shape // 2
    for x in np.ndindex(tuple(shape)):
        d[x] = np.linalg.norm((x - c) * dx)
    return d


def autocorr(x, ndim, dx=1, norm=None, pad=False):
    """
    http://www.jespertoftkristensen.com/JTK/Blog/Entries/2013/12/29_Efficient_computation_of_autocorrelations%3B_demonstration_in_MatLab_and_Python.html
    """
    norm = x.size if norm is None else norm
    x_ = np.pad(x, [(0, n) for n in x.shape], 'constant') if pad else x
    sp = fftn(x_)
    sq = sp * np.conj(sp) / norm
    autocor = np.real(fftshift(ifftn(sq[[slice(n) for n in x.shape]])))
    x = _get_distances(autocor.shape, dx=dx)
    return x, autocor


class StructureFFTAcc(object):
    def __init__(self, lattice_boxv, dx=1, sq_params=dict(norm=None), gr_params=dict(norm=None), nthreads=1):
        self.nthreads = nthreads
        self.lattice_shape = np.asarray(lattice_boxv)[::-1]
        self.ndim = len(self.lattice_shape)
        self.dx = dx
        self.sq_params = sq_params
        self.gr_params = gr_params
        self.freq = _get_freqs(self.lattice_shape, dx=self.dx)
        self.sq_acc = np.zeros(self.freq.shape)
        self.r = _get_distances(self.lattice_shape, dx=self.dx)
        self.gr_acc = np.zeros(self.r.shape)
        self.mean_acc = 0
        self.count = 0
        self.setup_fft()

    def setup_fft(self):
        self.a_ = pyfftw.empty_aligned(self.lattice_shape, dtype='complex128')
        self.b_ = pyfftw.empty_aligned(self.lattice_shape, dtype='complex128')
        self.ib_ = pyfftw.empty_aligned(self.lattice_shape, dtype='complex128')
        self.c_ = pyfftw.empty_aligned(self.lattice_shape, dtype='complex128')
        self.fft_obj = pyfftw.FFTW(self.a_, self.b_, flags=['FFTW_DESTROY_INPUT'],
                                   direction='FFTW_FORWARD', axes=range(self.ndim),
                                   threads=self.nthreads)
        self.ifft_obj = pyfftw.FFTW(self.ib_, self.c_, flags=['FFTW_DESTROY_INPUT'],
                                    direction='FFTW_BACKWARD', axes=range(self.ndim),
                                    threads=self.nthreads)

    def fftn(self, x):
        if x is not self.a_:
            np.copyto(self.a_, x)
        self.fft_obj()
        return self.b_

    def ifftn(self, x):
        """
        Backwards Real transform for the case in which the dimensionality
        of the transform is greater than 1 will destroy the input array
        my solution is to have another array ib into which to copy b, so
        that b does not get destroyed
        """
        if x is not self.ib_:
            np.copyto(self.ib_, x)
        self.ifft_obj()
        return self.c_

    def update(self, x_, clear=False):
        if clear:
            self.clear()
        x = x_ if (x_.shape == self.lattice_shape).all() else x_.reshape(self.lattice_shape)
        gr_norm = x.size if self.gr_params['norm'] is None else self.gr_params['norm']
        sq_norm = np.sum(x) if self.sq_params['norm'] is None else self.sq_params['norm']
        sp = self.fftn(x)
        sq_ = sp * np.conj(sp)
        gr = np.real(fftshift(self.ifftn(sq_))) / gr_norm
        sq = np.real(fftshift(sq_)) / sq_norm
        self.gr_acc += gr
        self.sq_acc += sq
        self.mean_acc += np.mean(x)
        self.count += 1

    def clear(self):
        self.sq_acc = np.zeros(self.freq.shape)
        self.gr_acc = np.zeros(self.r.shape)
        self.mean_acc = 0
        self.count = 0

    def get_gr(self):
        return self.r, self.gr_acc / self.count

    def get_sq(self):
        return self.freq, self.sq_acc / self.count

    def get_mean(self):
        return self.mean_acc / self.count

    def get_data(self):
        r, gr = self.get_gr()
        q, sq = self.get_sq()
        mean = self.get_mean()
        return dict(r=r, gr=gr, q=q, sq=sq, mean=mean)

    def __getstate__(self):
        """
        https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        """
        state = self.__dict__.copy()
        for k in ['a_', 'b_', 'ib_', 'c_', 'fft_obj', 'ifft_obj']:
            state.pop(k)
        return state

    def __setstate__(self, state):
        """
        https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        """
        self.__dict__.update(state)
        if 'nthreads' not in self.__dict__:  # for backwards compatibility
            self.nthreads = 1
        self.setup_fft()
