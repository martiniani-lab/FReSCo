import numpy as np
from collections import OrderedDict
from numba import jit


class ValueCounter(object):
    def __init__(self, init_count=0, init_value=0):
        self.count = init_count
        self.value = init_value

    def __add__(self, value):
        self.value += value
        self.count += 1
        return self

    def __str__(self):
        return "value: {:.3f}; count: {}".format(self.value, self.count)


class DistanceValueHistogram(object):
    """
    dist_value_count : 2D np.array
        list of lists containing [distance, value, count]
    """
    def __init__(self, rdecimals=9, dist_value_count=None):
        self.map = OrderedDict()
        self.rdecimals = rdecimals
        if dist_value_count is not None:
            for (d, v, c) in dist_value_count:
                self.update(d, v, c)

    def update(self, distance, value, count=1):
        # if rd in self.map:
        # else:
        rd = np.round(distance, self.rdecimals)
        try:
            self.map[rd] += value
        except KeyError:
            c = ValueCounter(count, value)
            self.map.update([(rd, c)])


    def _sort_map(self):
        self.map = OrderedDict(sorted(self.map.items()))

    def get_distance_values(self):
        self._sort_map()
        distances, normed_values = [], []
        for distance, value_count in self.map.iteritems():
            distances.append(distance)
            normed_values.append(value_count.value / value_count.count)
        return np.array(distances), np.array(normed_values)

    def get_distance_values_hist(self, nbins=20, dmin=None, dmax=None, mincount=10, geomspace=False, eps=1e-9):
        self._sort_map()
        distances, values, counts = [], [], []
        for distance, value_count in self.map.iteritems():
            distances.append(distance)
            values.append(value_count.value)
            counts.append(value_count.count)
        values, counts, distances = np.array(values), np.array(counts, dtype='int32'), np.array(distances)
        dmax = np.amax(distances) if dmax is None else dmax
        values = values[distances <= dmax]
        counts = counts[distances <= dmax]
        distances = distances[distances <= dmax]
        dmin = np.amin(distances) if dmin is None else dmin
        values = values[distances >= dmin]
        counts = counts[distances >= dmin]
        distances = distances[distances >= dmin]
        if geomspace:
            assert ((dmin + eps) > 0 and (dmax + eps > 0)), "geomspace needs dmin+eps > 0 and dmax-eps>0"
            bins = np.geomspace(dmin + eps, dmax + eps, nbins, endpoint=True, dtype='d')
        else:
            bins = np.linspace(dmin + eps, dmax + eps, nbins, endpoint=True, dtype='d')
        # digitize returns i such that bins[i-1] <= x < bins[i], that's why I shift evertything by eps=1e-9
        # then the mean value between bins and bins - eps is used -> (2. * bins - eps) / 2.
        inds = np.digitize(distances, bins)
        # print "inds: ", inds
        # print "distances: ", distances
        # print "bins: ", bins
        # print "bins[i]: ", bins[inds[0]], bins[inds[1]]
        # print np.amax(inds), np.amin(inds)
        histvalues = np.zeros(bins.size, dtype='d')
        histcounts = np.zeros(bins.size, dtype='d')
        for i, v, c in zip(inds, values, counts):
            histvalues[i] += v
            histcounts[i] += c
        # exclude points with poor statistics
        histvalues = histvalues[histcounts >= mincount]
        bins = bins[histcounts >= mincount]
        histcounts = histcounts[histcounts >= mincount]
        # print "hist bin counts: ", histcounts
        return (2. * bins - eps) / 2., histvalues / histcounts

    def __str__(self):
        stream = ""
        for dist, v in self.map.iteritems():
            stream += "dist: {:.3f}; {} \n".format(dist, v)
        return stream
