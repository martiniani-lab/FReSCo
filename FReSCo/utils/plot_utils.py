import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from itertools import cycle
from sklearn.covariance import MinCovDet, EllipticEnvelope
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm

def plot_circles_2d(file_name, points, radii, colors, L):
    N = len(radii)
    data = points.reshape(N,-1)
    ndim = data.shape[1]
    fig,ax = new_figure()
    colorcycle = get_color_cycle()
    for point,rad,col in zip(points,radii, colors):
        circ = plt.Circle(point,rad, color = colorcycle[col])
        ax.add_patch(circ)
    ax.set_xlim(0,L[0])
    ax.set_ylim(0,L[1])
    ax.set_aspect(L[1]/L[0])
    plt.axis('off')
    fig.set_size_inches(3,3)
    plt.savefig(file_name+'.png', bbox_inches='tight',dpi=100)

def new_figure(axarg='111'):
    fig = plt.figure()
    ax = fig.add_subplot('111')
    return fig, ax

def numerical_derivative(x, y, stencil=2):
    dy = np.zeros(y.size, np.float)
    if stencil == 2:
        dy[0:-1] = np.diff(y) / np.diff(x)
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    elif stencil == 3:
        for i in xrange(1, y.size - 1):
            dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        dy[0] = (y[1] - y[0]) / (x[1] - x[0])
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    elif stencil == 5:
        for i in xrange(2, y.size - 2):
            dy[i] = (y[i - 2] - 8 * y[i - 1] + 8 * y[i + 1] - y[i + 2]) / (3 * (x[i + 2] - x[i - 2]))
        dy[0] = (y[1] - y[0]) / (x[1] - x[0])
        dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        dy[1] = (y[2] - y[0]) / (x[2] - x[0])
        dy[-2] = (y[-1] - y[-3]) / (x[-1] - x[-3])
    else:
        raise NotImplementedError
    return x, dy


def fit_spline(x, y, w=None, bbox=[None, None], k=3, s=None, ext=0, check_finite=False):
    spl = UnivariateSpline(x, y, w=w, bbox=bbox, k=k, s=s,
                           ext=ext, check_finite=check_finite)
    return spl


def format_ticks_labels(ax, xaxis=True, yaxis=True, format="%.2f"):
    if xaxis:
        ax.xaxis.set_minor_formatter(FormatStrFormatter(format))
        ax.xaxis.set_major_formatter(FormatStrFormatter(format))
    if yaxis:
        ax.yaxis.set_minor_formatter(FormatStrFormatter(format))
        ax.yaxis.set_major_formatter(FormatStrFormatter(format))


def get_color_cycle(ncolors=8, reverse=True, cmap='Paired'):
    cm = plt.get_cmap(cmap)
    # cm = plt.get_cmap('Set3')
    if reverse:
        color_cycle = cycle([cm(1. * (i + 0.5) / float(ncolors)) for i in xrange(ncolors)][::-1])
    else:
        color_cycle = cycle([cm(1. * (i - 0.5) / float(ncolors)) for i in xrange(ncolors)])
    return color_cycle


def get_marker_cycle():
    markers = ["v", "o", "s", "^", ">", "<", "*", "h", "8", "p", "D"]
    markercycle = cycle(markers)
    return markercycle


def get_line_cycle(lines=["--", "-", ":", "-."]):
    linecycle = cycle(lines)
    return linecycle


def poly_fit(x, y, yerr=None, order=1):
    w = 1. / np.array(yerr) if yerr is not None else np.ones(len(x))
    x = np.asarray(x)
    X = np.column_stack(tuple([x ** i for i in xrange(1, order + 1)]))
    X = sm.add_constant(X)
    wls_model = sm.WLS(y, X, weights=w)
    results = wls_model.fit()
    fit_params = results.params[::-1]
    fit_err = results.bse[::-1]
    # print(results.summary())
    fit_fn = np.poly1d(fit_params)
    rho = pearsonr(x, y)[0]
    return fit_fn, fit_params, fit_err, rho


def exp_decay(x, y0, yinf, tau, delta):
    return (y0 - yinf) * np.exp(-x / tau) * np.power(x, -delta) + yinf


class ExpDecay(object):
    def __init__(self, x, y, tau, delta, y0=None, yinf=None, maxfev=3000):
        y0 = y[0] if y0 is None else y0
        yinf = y[-1] if yinf is None else yinf
        self.f = lambda x, y0, yinf, tau: exp_decay(x, y0, yinf, tau, delta)
        p0 = [y0, yinf, tau]
        self.popt, self.pcov = curve_fit(self.f, x, y, p0=p0, maxfev=maxfev)

    def __call__(self, x):
        return np.vectorize(self.f)(x, *self.popt)


def exp_decay2(x, y0, yinf, tau):
    return (y0 - yinf) * np.exp(-x / tau) * np.power(x + 1, -1. / 4) + yinf


class ExpDecay2(object):
    def __init__(self, x, y, y0, yinf, tau, maxfev=3000):
        self.f = lambda x, y0, yinf, tau: exp_decay2(x, y0, yinf, tau)
        p0 = [y0, yinf, tau]
        self.popt, self.pcov = curve_fit(self.f, x, y, p0=p0, maxfev=maxfev)

    def __call__(self, x):
        return np.vectorize(self.f)(x, *self.popt)


def sigmoid(x, x0, k, ymax, ymin, v):
    return ymax - (ymax - ymin) / np.power(1. + np.exp(-k * (x - x0)), 1. / v)


def sigmoid_d1(x, x0, k, ymax, ymin, v):
    return -(k / v) * (ymax - ymin) * np.exp(-k * (x - x0)) / np.power(1. + np.exp(-k * (x - x0)), 1. + 1. / v)


class Sigmoid(object):
    def __init__(self, x, y, p0, maxfev=3000):
        self.popt, self.pcov = curve_fit(sigmoid, x, y, p0=p0, maxfev=maxfev)

    def __call__(self, x):
        return np.vectorize(sigmoid)(x, *self.popt)

    def derivative(self):
        return lambda x: np.vectorize(sigmoid_d1)(x, *self.popt)


def sigmoid_fit(x, y, yerr=None, x0=0.84, k=1, ymax=None, ymin=None, v=1, maxfev=3000):
    ymax = np.amax(y) if ymax is None else ymax
    ymin = np.amin(y) if ymin is None else ymin
    popt, pcov = curve_fit(sigmoid, x, y, sigma=yerr, p0=[x0, k, ymax, ymin, v], maxfev=maxfev)
    fit_fn = lambda xx: np.vectorize(sigmoid)(xx, *popt)
    fit_fn_d1 = lambda xx: np.vectorize(sigmoid_d1)(xx, *popt)
    return fit_fn, fit_fn_d1, popt, pcov


def remove_outliers_mcd(x, y, yerr, support_fraction=0.99, contamination=0.1):
    x, y = np.asarray(x), np.asarray(y)
    classifier = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction, random_state=42)
    features = np.vstack((x, y)).T
    classifier.fit(features)
    decision = classifier.predict(features)
    mask = decision > 0
    return x[mask], y[mask], yerr[mask], mask


def rlm_fit(x, y, p0=None, order=1):
    assert len(x) == len(y)
    if order == 1:
        X = x
    elif order == 2:
        X = np.column_stack((x, x ** 2))
    elif order == 3:
        X = np.column_stack((x, x ** 2, x ** 3))
    else:
        raise NotImplementedError
    X = sm.add_constant(X)
    if p0 is None:
        p0 = np.ones(order + 1)
    assert len(p0) == order + 1
    model = sm.RLM(y, X)
    results = model.fit()
    return results, model


def lmms_fit(x, y, support_fraction=0.99):
    # linear minimum mean square error estimator
    x, y = np.asarray(x), np.asarray(y)
    robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42).fit(np.vstack((x, y)).T)
    cov = robust_cov.covariance_[0, 1]
    mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
    mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
    robust_rho = cov / np.sqrt(var_x * var_y)
    print(" robust rho", robust_rho)
    m = cov / var_x
    interc = mean_y - m * mean_x
    fit_fn = lambda xnew: m * (np.asarray(xnew) - mean_x) + mean_y
    # http://athenasc.com/Bivariate-Normal.pdf
    mserr = np.sqrt(var_y) * np.sqrt(1 - robust_rho ** 2)
    # https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression
    err = mserr ** 2 * np.array([[np.sum(x ** 2), -np.sum(x)], [-np.sum(x), x.size]]) / (
        x.size * np.sum(x ** 2) - np.sum(x) ** 2)
    fit_err = np.sqrt(np.diag(err))
    return fit_fn, (m, interc), fit_err[::-1], robust_rho


def lmms_fit_bs(x, y, support_fraction=0.99, reps=1000):
    # linear minimum mean square error estimator
    x, y = np.asarray(x), np.asarray(y)
    vars_bs = []
    robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42).fit(np.vstack((x, y)).T)
    cov = robust_cov.covariance_[0, 1]
    mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
    mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
    robust_rho = cov / np.sqrt(var_x * var_y)
    m = cov / var_x
    interc = mean_y - m * mean_x
    vars_bs.append([cov, mean_x, var_x, mean_y, var_y, robust_rho, m, interc])
    idx = np.random.choice(np.arange(len(x)), (reps, len(x)), replace=True)
    for i in xrange(reps - 1):
        robust_cov = MinCovDet(support_fraction=support_fraction,
                               random_state=42).fit(np.vstack((x[idx[i, :]], y[idx[i, :]])).T)
        cov = robust_cov.covariance_[0, 1]
        mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
        mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
        robust_rho = cov / np.sqrt(var_x * var_y)
        m = cov / var_x
        interc = mean_y - m * mean_x
        vars_bs.append([cov, mean_x, var_x, mean_y, var_y, robust_rho, m, interc])
    cov, mean_x, var_x, mean_y, var_y, robust_rho, m, interc = np.mean(vars_bs, axis=0)
    cov_se, mean_x_se, var_x_se, mean_y_se, var_y_se, robust_rho_se, m_se, interc_se = np.std(vars_bs, axis=0)
    fit_fn = lambda xnew: m * (np.asarray(xnew) - mean_x) + mean_y
    # http://athenasc.com/Bivariate-Normal.pdf
    fit_err = [m_se, interc_se]
    return fit_fn, (m, interc), fit_err, (robust_rho, robust_rho_se)


def robust_mean_var(x, y, support_fraction=0.99):
    x, y = np.asarray(x), np.asarray(y)
    robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42).fit(np.vstack((x, y)).T)
    cov = robust_cov.covariance_[0, 1]
    mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
    mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
    logl = robust_cov.score(np.vstack((x, y)).T)
    return (mean_x, var_x), (mean_y, var_y), cov, logl


def robust_mean_var_bs(x, y, support_fraction=0.99, reps=1000):
    x, y = np.asarray(x), np.asarray(y)
    vars_bs = []
    robust_cov = MinCovDet(support_fraction=support_fraction, random_state=42).fit(np.vstack((x, y)).T)
    cov = robust_cov.covariance_[0, 1]
    mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
    mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
    logl = robust_cov.score(np.vstack((x, y)).T)
    vars_bs.append([cov, mean_x, var_x, mean_y, var_y, logl])
    idx = np.random.choice(np.arange(len(x)), (reps, len(x)), replace=True)
    for i in xrange(reps - 1):
        robust_cov = MinCovDet(support_fraction=support_fraction,
                               random_state=42).fit(np.vstack((x[idx[i, :]], y[idx[i, :]])).T)
        cov = robust_cov.covariance_[0, 1]
        mean_x, var_x = robust_cov.location_[0], robust_cov.covariance_[0, 0]
        mean_y, var_y = robust_cov.location_[1], robust_cov.covariance_[1, 1]
        logl = robust_cov.score(np.vstack((x, y)).T)
        vars_bs.append([cov, mean_x, var_x, mean_y, var_y, logl])
    cov, mean_x, var_x, mean_y, var_y, logl = np.mean(vars_bs, axis=0)
    cov_se, mean_x_se, var_x_se, mean_y_se, var_y_se, logl_se = np.std(vars_bs, axis=0)
    return ((mean_x, var_x), (mean_y, var_y), cov, logl), (
        (mean_x_se, var_x_se), (mean_y_se, var_y_se), cov_se, logl_se)
