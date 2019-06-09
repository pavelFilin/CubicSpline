import typing as t
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

_BreaksDataType = t.Union[
    # Univariate data sites
    np.ndarray,

    # Grid data sites
    t.Union[
        t.List[np.ndarray],
        t.Tuple[np.ndarray, ...]
    ]
]

_UnivariateDataType = t.Union[
    np.ndarray,
    t.Sequence[t.Union[int, float]]
]

_UnivariateVectorizedDataType = t.Union[
    _UnivariateDataType,
    t.List['_UnivariateVectorizedDataType']
]

_MultivariateData = t.Union[
    np.ndarray,
    t.Sequence[_UnivariateDataType]
]

_GridDataType = t.Sequence[_UnivariateDataType]


class SplinePPForm:
    def __init__(self, breaks: _BreaksDataType, coeffs: np.ndarray, dim: int = 1):
        self.gridded = isinstance(breaks, (tuple, list))
        self.breaks = breaks
        self.coeffs = coeffs
        self.pieces = None  # type: t.Union[int, t.Tuple[int, ...]]
        self.order = None  # type: t.Union[int, t.Tuple[int, ...]]

        if self.gridded:
            self.pieces = tuple(x.size - 1 for x in breaks)
            self.order = tuple(s // p for s, p in zip(coeffs.shape[1:], self.pieces))
            self.dim = len(breaks)
        else:
            self.pieces = np.prod(coeffs.shape[:-1]) // dim
            self.order = coeffs.shape[-1]
            self.dim = dim

    def __str__(self):
        return (
            '{}\n'
            '  gridded: {}\n'
            '  breaks: {}\n'
            '  coeffs: {}\n{}\n'
            '  pieces: {}\n'
            '  order: {}\n'
            '  dim: {}\n'
        ).format(self.__class__.__name__,
                 self.gridded, self.breaks, self.coeffs.shape, self.coeffs,
                 self.pieces, self.order, self.dim)

    def evaluate(self, xi, shape=None):
        if self.gridded:
            return self._grid_evaluate(xi)
        else:
            return self._univariate_evaluate(xi, shape)

    def _univariate_evaluate(self, xi, shape):
        mesh = self.breaks[1:-1]
        edges = np.hstack((-np.inf, mesh, np.inf))
        index = np.digitize(xi, edges)
        nanx = np.flatnonzero(index == 0)
        index = np.fmin(index, mesh.size + 1)
        index[nanx] = 1
        xi = xi - self.breaks[index - 1]
        d = self.dim
        lx = len(xi)
        if d > 1:
            xi_shape = (1, d * lx)
            xi_ndm = np.array(xi, ndmin=2)
            xi = np.reshape(np.repeat(xi_ndm, d, axis=0), xi_shape, order='F')

            index_rep = (np.repeat(np.array(1 + d * index, ndmin=2), d, axis=0)
                         + np.repeat(np.array(np.r_[-d:0], ndmin=2).T, lx, axis=1))
            index = np.reshape(index_rep, (d * lx, 1), order='F')
        index -= 1
        values = self.coeffs[index, 0].T
        for i in range(1, self.coeffs.shape[1]):
            values = xi * values + self.coeffs[index, i].T
        values = values.reshape((d, lx), order='F').squeeze()
        if values.shape != shape:
            values = values.reshape(shape)
        return values

    def _grid_evaluate(self, xi):
        yi = self.coeffs.copy()
        sizey = list(yi.shape)
        nsize = tuple(x.size for x in xi)

        for i in range(self.dim - 1, -1, -1):
            dim = int(np.prod(sizey[:self.dim]))
            coeffs = yi.reshape((dim * self.pieces[i], self.order[i]), order='F')

            spp = SplinePPForm(self.breaks[i], coeffs, dim=dim)
            yi = spp.evaluate(xi[i], shape=(dim, xi[i].size))

            yi = yi.reshape((*sizey[:self.dim], nsize[i]), order='F')
            axes = (0, self.dim, *np.r_[1:self.dim].tolist())
            yi = yi.transpose(axes)
            sizey = list(yi.shape)

        return yi.reshape(nsize, order='F')


class CubicSmoothingSpline:
    def __init__(self,
                 xdata: _UnivariateDataType,
                 ydata: _UnivariateVectorizedDataType,
                 weights: t.Optional[_UnivariateDataType] = None,
                 smooth: t.Optional[float] = None):

        self._spline: SplinePPForm = None
        self._smooth = smooth

        (self._xdata,
         self._ydata,
         self._weights,
         self._data_shape) = self._prepare_data(xdata, ydata, weights)

        self._ydim = self._ydata.shape[0]
        self._axis = self._ydata.ndim - 1

        self._make_spline()

    def __call__(self, xi: _UnivariateDataType) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        if xi.ndim > 1:
            raise ValueError('XI data must be a vector.')
        self._data_shape[-1] = xi.size
        return self._spline.evaluate(xi, self._data_shape)

    @property
    def smooth(self) -> float:
        return self._smooth

    @property
    def spline(self) -> SplinePPForm:
        return self._spline

    @staticmethod
    def _prepare_data(xdata, ydata, weights):
        xdata = np.asarray(xdata, dtype=np.float64)
        ydata = np.asarray(ydata, dtype=np.float64)

        data_shape = list(ydata.shape)

        if xdata.ndim > 1:
            raise ValueError('xdata must be a vector')
        if xdata.size < 2:
            raise ValueError('xdata must contain at least 2 data points.')

        if ydata.ndim > 1:
            if data_shape[-1] != xdata.size:
                raise ValueError(
                    'ydata data must be a vector or '
                    'ND-array with shape[-1] equal of xdata.size')
            if ydata.ndim > 2:
                ydata = ydata.reshape((np.prod(data_shape[:-1]), data_shape[-1]))
        else:
            if ydata.size != xdata.size:
                raise ValueError('ydata vector size must be equal of xdata size')

            ydata = np.array(ydata, ndmin=2)

        if weights is None:
            weights = np.ones_like(xdata)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.size != xdata.size:
                raise ValueError(
                    'Weights vector size must be equal of xdata size')

        return xdata, ydata, weights, data_shape

    @staticmethod
    def _compute_smooth(a, b):
        def trace(m: sp.dia_matrix):
            return m.diagonal().sum()
        return 1. / (1. + trace(a) / (6. * trace(b)))

    def _make_spline(self):
        pcount = self._xdata.size

        dx = np.diff(self._xdata)

        if not all(dx > 0):
            raise ValueError(
                'Items of xdata vector must satisfy the condition: x1 < x2 < ... < xN')

        dy = np.diff(self._ydata, axis=self._axis)
        divdydx = dy / dx

        if pcount > 2:
            diags_r = np.vstack((dx[1:], 2 * (dx[1:] + dx[:-1]), dx[:-1]))
            r = sp.spdiags(diags_r, [-1, 0, 1], pcount - 2, pcount - 2)

            odx = 1. / dx
            diags_qt = np.vstack((odx[:-1], -(odx[1:] + odx[:-1]), odx[1:]))
            qt = sp.diags(diags_qt, [0, 1, 2], (pcount - 2, pcount))

            ow = 1. / self._weights
            osqw = 1. / np.sqrt(self._weights)  # type: np.ndarray
            w = sp.diags(ow, 0, (pcount, pcount))
            qtw = qt @ sp.diags(osqw, 0, (pcount, pcount))
            qtwq = qtw @ qtw.T

            if self._smooth:
                p = self._smooth
            else:
                p = self._compute_smooth(r, qtwq)

            a = (6. * (1. - p)) * qtwq + p * r
            b = np.diff(divdydx, axis=self._axis).T
            u = np.array(la.spsolve(a, b), ndmin=2)

            if self._ydim == 1:
                u = u.T

            dx = np.array(dx, ndmin=2).T
            d_pad = np.zeros((1, self._ydim))

            d1 = np.diff(np.vstack((d_pad, u, d_pad)), axis=0) / dx
            d2 = np.diff(np.vstack((d_pad, d1, d_pad)), axis=0)

            yi = np.array(self._ydata, ndmin=2).T
            yi = yi - ((6. * (1. - p)) * w) @ d2
            c3 = np.vstack((d_pad, p * u, d_pad))
            c2 = np.diff(yi, axis=0) / dx - dx * (2. * c3[:-1, :] + c3[1:, :])

            coeffs = np.hstack((
                (np.diff(c3, axis=0) / dx).T,
                3. * c3[:-1, :].T,
                c2.T,
                yi[:-1, :].T
            ))

            c_shape = ((pcount - 1) * self._ydim, 4)
            coeffs = coeffs.reshape(c_shape, order='F')
        else:
            p = 1.
            coeffs = np.array(np.hstack(
                (divdydx, np.array(self._ydata[:, 0], ndmin=2).T)), ndmin=2)

        self._smooth = p
        self._spline = SplinePPForm(self._xdata, coeffs, self._ydim)

