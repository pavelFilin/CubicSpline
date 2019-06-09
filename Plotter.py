import numpy as np
import matplotlib.pyplot as plt
import CubicSpline

np.random.seed(1234)

x = np.linspace(-50., 50., 30)
x2 = np.arange(-50, 50, 0.2)
y2 = np.exp(-(x2 / 2.5) ** 2)
y = np.exp(-(x / 2.5) ** 2)

sp = CubicSpline.CubicSmoothingSpline(x, y, smooth=0.9)

xs = np.linspace(x[0], x[-1], 300)
ys = sp(xs)

plt.plot(x, y, 'o', xs, ys, '-', label="cubic spline")
plt.plot(x2, y2, '-', label="cubic spline")

plt.show()
