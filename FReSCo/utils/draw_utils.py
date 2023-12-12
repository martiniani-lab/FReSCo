import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from FReSCo.distances import put_in_box, Distance, get_distance, put_atom_in_box
from PIL import Image

def plot_points(file_name, points, L):
    p = np.reshape(points, (-1, len(L))) % L[0]
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(p[:, 0], p[:, 1])
    ax.set_xlim(0, L[0])
    ax.set_ylim(L[1], 0)
    ax.set_aspect(aspect=1.0)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(file_name + '.png', bbox_inches='tight',
                pad_inches=0, dpi=300)
    plt.close()

def plot_voro(file_name, points, L):
    p = np.reshape(points, (-1, len(L))) % L[0]
    vor = Voronoi(p)
    fig = voronoi_plot_2d(vor, show_vertices=False)
    ax = fig.gca()
    ax.set_xlim(0, L[0])
    ax.set_ylim(L[1], 0)
    ax.set_aspect(aspect=1.0)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(file_name + '_voro.png', bbox_inches='tight',
                pad_inches=0, dpi=300)
    plt.close()

def flip(m, axis):
    # https://github.com/numpy/numpy/blob/v1.13.0/numpy/lib/function_base.py#L141-L210
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

def write_opengl_input(coords, radii, boxv, distance_method=Distance.PERIODIC):
    """write opengl input file"""
    ndim = len(boxv)
    coords = put_in_box(coords, ndim, distance_method, boxv)
    colour = 13
    directory = os.getcwd()
    bdim = len(boxv)
    nparticles = radii.size
    fname = "{0}/packing.dat".format(directory)
    f = open(fname, 'w')
    f.write('{}\n'.format(nparticles))

    if bdim == 2:
        f.write('{} {} {}\n'.format(-boxv[0] / 2, -boxv[1] / 2,
                                    -np.amax(radii)))
        f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
        f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
        f.write('0.0 \t 0.0 \t {}\n'.format(np.amax(radii) * 2))
        for i in xrange(nparticles):
            for j in xrange(bdim):
                f.write('{}\t'.format(coords[i * bdim + j]))
            f.write('{}\t'.format(0))
            f.write('{}\t'.format(radii[i] * 2))
            f.write('{}\n'.format(colour))
    elif bdim == 3:
        f.write('{} {} {}\n'.format(-boxv[0] / 2, -boxv[1] / 2, -boxv[2] / 2))
        f.write('{} \t 0.0 \t 0.0\n'.format(boxv[0]))
        f.write('0.0 \t {} \t 0.0\n'.format(boxv[1]))
        f.write('0.0 \t 0.0 \t {}\n'.format(boxv[2]))
        for i in xrange(nparticles):
            for j in xrange(bdim):
                f.write('{}\t'.format(coords[i * bdim + j]))
            f.write('{}\t'.format(radii[i] * 2))
            f.write('{}\n'.format(colour))
    else:
        raise NotImplementedError("bdim={} not implemented"
                                  .format(bdim))
    f.close()


def draw_velocity_vectors(coords, velocities, boxv, distance_method=Distance.PERIODIC, ax=None):
    ndim = len(boxv)
    assert ndim == 2
    if distance_method == Distance.PERIODIC:
        coords = put_in_box(coords, ndim, distance_method, boxv)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    X, Y = coords[0::2], coords[1::2]
    U, V = velocities[0::2], velocities[1::2]
    ax.quiver(X, Y, U, V, color='r')
    return ax

def draw_circles(coords, radii, boxv, distance_method=Distance.PERIODIC, label=False, ax=None, alpha=0.5):
    ndim = len(boxv)
    if distance_method == Distance.PERIODIC:
        coords = put_in_box(coords, ndim, distance_method, boxv)
    if distance_method == Distance.PERIODIC:
        x = np.array(coords).reshape((-1, 2))
        x_ = np.array(x)
        x_[:, 0] += boxv[0]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 0] -= boxv[0]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 1] += boxv[1]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 1] -= boxv[1]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 0] += boxv[0]
        x_[:, 1] += boxv[1]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 0] -= boxv[0]
        x_[:, 1] -= boxv[1]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 0] += boxv[0]
        x_[:, 1] -= boxv[1]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        x_ = np.array(x)
        x_[:, 0] -= boxv[0]
        x_[:, 1] += boxv[1]
        coords = np.append(coords, x_)
        radii = np.append(radii, radii)
        coords = coords.ravel()
        radii = radii.ravel()
    coords = np.array(coords).reshape(-1, ndim)
    patches = []
    for i, (x, r) in enumerate(zip(coords, radii)):
        circle = Circle(x, radius=r, alpha=alpha, color='b') #edgecolor='b'
        patches.append(circle)
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if label:
        for i, (patch, x) in enumerate(zip(patches, coords)):
            ax.add_patch(patch, rasterized=True)
            ax.annotate(i, x)
    else:
        p = PatchCollection(patches, match_original=True, rasterized=True)
        ax.add_collection(p)
    ax.set_aspect('equal')
    if distance_method == Distance.PERIODIC:
        ax.set_xlim(-boxv[0] / 2, boxv[0] / 2)
        ax.set_ylim(-boxv[1] / 2, boxv[1] / 2)
    else:
        ax.set_xlim(0, boxv[0])
        ax.set_ylim(0, boxv[1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return fig, ax


def draw_lattice2d(lattice, lattice_boxv, flipim=True, cmap='Greys', interpolation='nearest', **kwargs):
    # lattice shape is (lx, ly, lx) but the matrix shape is (lz, ly, lx)
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax, im = _draw_lattice2d(ax, lattice, lattice_boxv, flipim, cmap, interpolation, **kwargs)
    cbar = fig.colorbar(im)
    # im = ax.matshow(im_lattice)
    return fig, ax, im, cbar


def _draw_lattice2d(ax, lattice, lattice_boxv, flipim=True, cmap='Greys', interpolation='none', **kwargs):
    # lattice shape is (lx, ly, lx) but the matrix shape is (lz, ly, lx)
    ndim = len(lattice_boxv)
    assert ndim == 2
    lattice_shape = lattice_boxv[::-1]
    ax.set_aspect('equal', adjustable='box')
    if flipim:
        im_lattice = flip(lattice.reshape(lattice_shape), axis=0)
    else:
        im_lattice = lattice.reshape(lattice_shape)
    im = ax.imshow(im_lattice, cmap=cmap, interpolation=interpolation, **kwargs)
    return ax, im


def draw_lattice_fft(lattice, lattice_boxv, cmap='gray', lambdaf=None, ax=None):
    ndim = len(lattice_boxv)
    assert ndim == 2
    lattice_shape = lattice_boxv[::-1]
    im_lattice = flip(lattice.reshape(lattice_shape), axis=0)
    f = np.fft.fft2(im_lattice)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) if lambdaf is None else lambdaf(np.abs(fshift))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot('111')
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(magnitude_spectrum, cmap=cmap)
    return im


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    # now convert to greyscale
    w, h, d = buf.shape
    print(w, h)
    img = Image.frombytes("RGBA", (w, h), buf.tostring()).convert('L')
    img_array = np.asarray(img.getdata()).reshape(img.size)
    return img, img_array


def draw_fft(img_array, cmap='gray', lambdaf=None):
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift) if lambdaf is None else lambdaf(np.abs(fshift))
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.set_aspect('equal', adjustable='box')
    im = ax.imshow(magnitude_spectrum, cmap=cmap)
    return im
