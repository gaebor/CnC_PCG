import numpy
import math
from numpy import random
from perlin2d import *

import zlib

"""
   +-+
   | |
   +-+
   
   "+": node
   "-", "|": edge
   " ": face, tile, cell
"""


def perlin(shape0, shape1, **kwargs):
    n = 2 ** math.ceil(numpy.log2(max(shape0, shape1)))
    octaves = 1 + int(numpy.log2(n))
    res = (1, 1)
    X = generate_fractal_noise_2d((n, n), res, octaves)[:shape0, :shape1]
    Y = generate_fractal_noise_2d((n, n), res, octaves)[:shape0, :shape1]
    return X, Y


def simple(n, m, H=0.5):
    """http://paulbourke.net/fractals/noise/"""
    X = numpy.random.standard_normal(size=(n, m)) + 1j * numpy.random.standard_normal(size=(n, m))
    r = numpy.sqrt(
        (numpy.linspace(1, n, n) ** 2)[:, None] + (numpy.linspace(1, m, m) ** 2)[None, :]
    )
    F = numpy.fft.fft2(X / r ** (4 * H))
    return numpy.real(F), numpy.imag(F)


def brownian(n, m, H=0.5):
    """
    Stochastic Geometry, Spatial Statistics and Random Fields Models and Algorithms,
    Section: Generating Stationary Processes via Circulant Embedding
    """
    alpha = 2 * H
    R = 2
    if alpha <= 1.5:
        beta = 0
        c2 = alpha / 2
        c0 = 1 - alpha / 2
    else:
        beta = alpha * (2 - alpha) / (3 * R * (R ** 2 - 1))
        c2 = (alpha - beta * (R - 1) ** 2 * (R + 2)) / 2
        c0 = beta * (R - 1) ** 3 + 1 - c2

    def rho(r):
        out = beta * (R - r) ** 3 / r
        out[r <= 1] = 0
        out += (r <= 1) * (c0 - r ** alpha + c2 * r ** 2)
        out[r > R] = 0
        return out

    tx = numpy.arange(n) / n * R
    ty = numpy.arange(m) / m * R
    Rows = rho(numpy.sqrt((tx ** 2)[:, None] + ((ty ** 2)[None, :])))
    BlkCirc_row = numpy.block(
        [[Rows, Rows[:, -2:0:-1]], [Rows[-2:0:-1, :], Rows[-2:0:-1, -2:0:-1]]]
    )
    lam = numpy.sqrt(numpy.real(numpy.fft.fft2(BlkCirc_row) / (4 * (m - 1) * (n - 1))))
    Z = random.standard_normal(size=(2 * (n - 1), 2 * (m - 1))) + 1j * random.standard_normal(
        size=(2 * (n - 1), 2 * (m - 1))
    )
    F = numpy.fft.fft2(lam * Z)[:n, :m]
    F -= F[0, 0]
    for i in [numpy.sqrt(2 * c2), 1j * numpy.sqrt(2 * c2)]:
        F += i * (
            (tx * random.standard_normal())[:, None] * (ty * random.standard_normal())[None, :]
        )

    return numpy.real(F), numpy.imag(F)


def highest_bit(x, dtype='int32'):
    # TODO ufunc
    y = (numpy.log2(x) + 1).astype(dtype)
    y[x == 0] = 0
    y[x < 0] = numpy.dtype(dtype).itemsize * 8
    return y


def find_contours(F, dhbase=0.125):
    """
    Contour (edge) is formed between two different height cells.
    The height difference determines the numbering of the contours through a 
    
    'f(height1, height2) -> edge number' function.
    It's the highest 2 power between the height of the lines.
    
    between neighboring numbers:
    0   1   2   3   4   5   6   7   8   9   10
     \ / \ / \ / \ / \ / \ / \ / \ / \ / \ /
      1   2   1   3   1   2   1   4   1   2
    
    in formula: 'highest_bit(x XOR y)'
    
    water is -1 and if a contour meets water then it has higher hierarchy then anything on land.
    """
    E = numpy.zeros((2 * F.shape[0] - 1, 2 * F.shape[1] - 1), 'int32')
    E[::2, ::2] = numpy.maximum(-1, numpy.floor(F / dhbase).astype('int32'))

    E[1::2, ::2] = highest_bit(E[2::2, ::2] ^ E[:-2:2, ::2])
    E[::2, 1::2] = highest_bit(E[::2, 2::2] ^ E[::2, :-2:2])

    return E


def hierarchical_contours(E, minh=0):
    """
    Deletes every lower hierarchy contour in the neighborhood of any higher hierarchy contours.
    From highest hierarchy down to lowest level hierarchy.
    This ensures that no two different level contours meet
    (same level contours can meet because they are just level-lines of a surface, therefore consistent*).
    In case of crossing, the higher level contours is continued and lower level is broken up.
    
    The remaining contours should be kept consistent (no T junctions)
    therefore a higher level contour deletes the lower level contours around itself.
    
    TODO: what did I do here? It was the key somehow.
    You cannot delete any edge, you have to delete a whole cell to keep the remaining part
    consistent. By that I mean no dangling edges.
    
    good:
     _
    |_|_
    |_|_|
    
    bad:
     _
    |_|_
    |_|_
        
    """
    h = 32
    while h > 0:
        E[::2, 1:-2:2][numpy.logical_and(E[::2, 1:-2:2] < h, E[::2, 3::2] == h)] = 0
        E[::2, 3::2][numpy.logical_and(E[::2, 3::2] < h, E[::2, 1:-2:2] == h)] = 0

        E[1::2, :-2:2][numpy.logical_and(E[1::2, :-2:2] < h, E[:-1:2, 1::2] == h)] = 0
        E[1::2, 2::2][numpy.logical_and(E[1::2, 2::2] < h, E[:-1:2, 1::2] == h)] = 0

        E[1::2, :-2:2][numpy.logical_and(E[1::2, :-2:2] < h, E[2::2, 1::2] == h)] = 0
        E[1::2, 2::2][numpy.logical_and(E[1::2, 2::2] < h, E[2::2, 1::2] == h)] = 0

        E[1:-2:2, ::2][numpy.logical_and(E[1:-2:2, ::2] < h, E[3::2, ::2] == h)] = 0
        E[3::2, ::2][numpy.logical_and(E[3::2, ::2] < h, E[1:-2:2, ::2] == h)] = 0

        E[:-2:2, 1::2][numpy.logical_and(E[:-2:2, 1::2] < h, E[1::2, :-1:2] == h)] = 0
        E[2::2, 1::2][numpy.logical_and(E[2::2, 1::2] < h, E[1::2, :-1:2] == h)] = 0

        E[:-2:2, 1::2][numpy.logical_and(E[:-2:2, 1::2] < h, E[1::2, 2::2] == h)] = 0
        E[2::2, 1::2][numpy.logical_and(E[2::2, 1::2] < h, E[1::2, 2::2] == h)] = 0

        h -= 1

    if type(minh) == numpy.ndarray:
        E[1::2, ::2][E[1::2, ::2] <= minh[1:, :]] = 0
        E[1::2, ::2][E[1::2, ::2] <= minh[:-1, :]] = 0

        E[::2, 1::2][E[::2, 1::2] <= minh[:, 1:]] = 0
        E[::2, 1::2][E[::2, 1::2] <= minh[:, :-1]] = 0
    else:
        E[1::2, ::2][E[1::2, ::2] <= minh] = 0
        E[::2, 1::2][E[::2, 1::2] <= minh] = 0

    return E


def threshold_contours(E, dx):
    # must erase marked contours on land and their neighboring edges
    # TODO: may erase marked contours on beach

    pattern = numpy.random.rand((E.shape[0] + 1) // 2, (E.shape[1] + 1) // 2) < dx
    pattern[E[::2, ::2] < 0] = False
    (pattern[:, 1:])[E[::2, :-1:2] < 0] = False
    (pattern[:, :-1])[E[::2, 2::2] < 0] = False
    (pattern[1:, :])[E[:-1:2, ::2] < 0] = False
    (pattern[:-1, :])[E[2::2, ::2] < 0] = False

    E[1::2, ::2][pattern[1:, :]] = 0
    E[1::2, ::2][pattern[:-1, :]] = 0

    E[::2, 1::2][pattern[:, 1:]] = 0
    E[::2, 1::2][pattern[:, :-1]] = 0
    return E


def check3(E):
    """T junctions"""
    return numpy.where(
        (
            (E[:-2:2, 1::2] > 0).astype("int32")
            + (E[2::2, 1::2] > 0).astype("int32")
            + (E[1::2, :-2:2] > 0).astype("int32")
            + (E[1::2, 2::2] > 0).astype("int32")
        )
        == 3
    )


def generate_map(F, dx=0, dhbase=0.125, dh=1):
    """https://forums.cncnet.org/topic/4769-automatical-landscape-generation/"""
    E = find_contours(F, dhbase)
    hierarchical_contours(E, dh)
    threshold_contours(E, dx)
    return E

