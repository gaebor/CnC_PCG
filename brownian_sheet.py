import numpy
from numpy import random
from perlin2d import *

import zlib
# numpy.set_printoptions(linewidth=180)

"""
   +-+
   | |
   +-+
   
   "+": node
   "-", "|": edge
   " ": face, tile, cell
"""

def linear_estimate(x,y):
    a, Ba = x
    b, Bb = y
    c = (a+b)//2
    return (c, 0.5 * numpy.linalg.norm(a-b)**0.5 * fixed_normal(c))

def parallel(a,b):
    return a[0]*b[1] - a[1]*b[0] == 0

def perpendicular(a,b):
    return a[0]*b[0] + a[1]*b[1] == 0
    
def sameside(a,b,c,x):
    s = b-a
    s = s[[1,0]]
    s[1] *= -1
    v = numpy.array([x-a,c-a]).dot(s)
    return numpy.sign(v).prod()

def descent(x, n=3):
    sqrta = 2**(n/2)
    l = (numpy.array([0,0], dtype='int32'), 
         numpy.array([2**n,0], dtype='int32'),
         numpy.array([0,2**n], dtype='int32'))
    v = (0.0,)
    v += (sqrta * fixed_normal(l[1]),)
    v += (0.2928932188134524 * v[1] + 0.9561451575849219 * sqrta * fixed_normal(l[2]),)
    print(v)
    result = [l]
    while len(l) > 1:
        if any((a==x).all() for a in l):
            l = (x,)
        elif len(l) == 2:
            y = (l[0] + l[1])//2 # TODO calculate
            if numpy.linalg.norm(l[0]-x) < numpy.linalg.norm(l[1]-x):
                l = (l[0], y)
            else:
                l = (l[1], y)
        else:
            a = l[0]
            b = l[1]
            c = l[2]
            if parallel(x-a, b-a):
                l = (a, b)
            elif parallel(x-b, c-b):
                l = (b, c)
            elif parallel(x-a, c-a):
                l = (a, c)
            else:
                # TODO calculate
                ab = (a+b)//2
                ac = (a+c)//2
                bc = (b+c)//2
                if sameside(ab, ac, a, x) >= 0:
                    l = (a, ab, ac)
                elif sameside(ab, bc, b, x) >= 0:
                    l = (b, ab, bc)
                else:
                    l = (c, ac, bc)
        result.append(l)
        print(l)
    return result

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

def generate_perlin(shape, res=(1,1), octaves=1, H=0.5):
    noise = numpy.zeros(shape)
    frequency = 1
    for n in range(1, octaves+1):
        noise += ((1/n)**H)*generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[0]))
        frequency *= 2
    return noise

def fixed_normal(x, size=None):
    random.seed(zlib.crc32(repr(x).encode("utf-8")) & 0xffffffff)
    return random.standard_normal(size)

def generate(n, m, H=0.5):
    """
    Stochastic Geometry, Spatial Statistics and Random Fields Models and Algorithms,
    Section: Generating Stationary Processes via Circulant Embedding
    """
    alpha = 2*H
    R = 2
    if alpha <= 1.5:
        beta = 0
        c2 = alpha/2
        c0 = 1-alpha/2
    else:
        beta = alpha*(2-alpha)/(3*R*(R**2-1))
        c2 = (alpha-beta*(R-1)**2 * (R+2))/2
        c0 = beta*(R-1)**3+1-c2
        
    def rho(r):
        out = beta*(R-r)**3/r
        out[r<=1] = 0
        out += (r<=1)*(c0-r**alpha+c2*r**2)
        out[r>R] = 0
        return out
    
    tx=numpy.arange(n)/n*R
    ty=numpy.arange(m)/m*R
    Rows=rho(numpy.sqrt((tx**2)[:, None] + ((ty**2)[None, :])))
    BlkCirc_row=numpy.block([[Rows,            Rows[:,-2:0:-1]      ],
                             [Rows[-2:0:-1,:], Rows[-2:0:-1,-2:0:-1]]])
    lam = numpy.sqrt(numpy.real(numpy.fft.fft2(BlkCirc_row)/(4*(m-1)*(n-1))))
    Z = random.standard_normal(size=(2*(n-1),2*(m-1))) + 1j*random.standard_normal(size=(2*(n-1),2*(m-1)))
    F = numpy.fft.fft2(lam*Z)[:n, :m]
    F -= F[0,0]
    for i in [numpy.sqrt(2*c2), 1j*numpy.sqrt(2*c2)]:
        F += i*((tx*random.standard_normal())[:, None] * (ty*random.standard_normal())[None, :])
    
    return numpy.real(F), numpy.imag(F)

def highest_bit(x, dtype='int32'):
    # TODO ufunc
    y = (numpy.log2(x)+1).astype(dtype)
    y[x == 0] = 0
    y[x < 0]  = numpy.dtype(dtype).itemsize * 8
    return y

def find_contours(F, dhbase=0.125):
    E = numpy.zeros((2*F.shape[0]-1, 2*F.shape[1]-1), 'int32')
    E[::2, ::2] = numpy.maximum(-1, numpy.floor(F/dhbase).astype('int32'))
    
    E[1::2, ::2] = highest_bit(E[2::2, ::2] ^ E[:-2:2, ::2])
    E[::2, 1::2] = highest_bit(E[::2, 2::2] ^ E[::2, :-2:2])
    
    return E

def hierarchical_contours(E, minh=0):
    h = 32
    while h > 0:
        E[ :  :2, 1:-2:2][numpy.logical_and(E[ :  :2, 1:-2:2] < h, E[:  :2, 3:  :2] == h)] = 0
        E[ :  :2, 3:  :2][numpy.logical_and(E[ :  :2, 3:  :2] < h, E[:  :2, 1:-2:2] == h)] = 0
        
        E[1:  :2,  :-2:2][numpy.logical_and(E[1:  :2,  :-2:2] < h, E[:-1:2, 1:  :2] == h)] = 0
        E[1:  :2, 2:  :2][numpy.logical_and(E[1:  :2, 2:  :2] < h, E[:-1:2, 1:  :2] == h)] = 0
        
        E[1:  :2,  :-2:2][numpy.logical_and(E[1:  :2,  :-2:2] < h, E[2: :2, 1:  :2] == h)] = 0
        E[1:  :2, 2:  :2][numpy.logical_and(E[1:  :2, 2:  :2] < h, E[2: :2, 1:  :2] == h)] = 0

        E[ 1:-2:2, :  :2][numpy.logical_and(E[ 1:-2:2, :  :2] < h, E[ 3:  :2,:  :2] == h)] = 0
        E[ 3:  :2, :  :2][numpy.logical_and(E[ 3:  :2, :  :2] < h, E[ 1:-2:2,:  :2] == h)] = 0
        
        E[  :-2:2,1:  :2][numpy.logical_and(E[  :-2:2,1:  :2] < h, E[ 1:  :2,:-1:2] == h)] = 0
        E[ 2:  :2,1:  :2][numpy.logical_and(E[ 2:  :2,1:  :2] < h, E[ 1:  :2,:-1:2] == h)] = 0
        
        E[  :-2:2,1:  :2][numpy.logical_and(E[  :-2:2,1:  :2] < h, E[ 1:  :2,2: :2] == h)] = 0
        E[ 2:  :2,1:  :2][numpy.logical_and(E[ 2:  :2,1:  :2] < h, E[ 1:  :2,2: :2] == h)] = 0
        
        h -= 1

    if type(minh) == numpy.ndarray:
        E[1::2, ::2][E[1::2, ::2] <= minh[1:,:]] = 0
        E[1::2, ::2][E[1::2, ::2] <= minh[:-1,:]] = 0
        
        E[::2, 1::2][E[::2, 1::2] <= minh[:,1:]] = 0
        E[::2, 1::2][E[::2, 1::2] <= minh[:,:-1]] = 0
    else:
        E[1::2, ::2][E[1::2, ::2] <= minh] = 0
        E[::2, 1::2][E[::2, 1::2] <= minh] = 0
    
    return E

def threshold_contours(E, dx):
    # must erase marked contours on land and their neighboring edges
    # TODO: may erase marked contours on beach
    
    pattern = numpy.random.rand((E.shape[0]+1)//2, (E.shape[1]+1)//2) < dx
    pattern[E[::2,::2] < 0] = False
    (pattern[:, 1:])[E[::2,:-1:2] < 0] = False
    (pattern[:, :-1])[E[::2,2::2] < 0] = False
    (pattern[1:, :])[E[:-1:2,::2] < 0] = False
    (pattern[:-1, :])[E[2::2,::2] < 0] = False
    
    E[1::2, ::2][ pattern[1:,:] ] = 0
    E[1::2, ::2][ pattern[:-1,:] ] = 0
    
    E[::2, 1::2][ pattern[:,1:] ] = 0
    E[::2, 1::2][ pattern[:,:-1] ] = 0
    return E

def check3(E):
    """T junctions"""
    return numpy.where((
        (E[:-2:2, 1::2] > 0).astype("int32") + (E[2::2, 1::2] > 0).astype("int32") + \
        (E[1::2, :-2:2] > 0).astype("int32") + (E[1::2, 2::2] > 0).astype("int32")
                        ) == 3)

def generate_map(F, dx=0, dhbase=0.125, dh=1):
    """https://forums.cncnet.org/topic/4769-automatical-landscape-generation/"""
    E = find_contours(F, dhbase)
    hierarchical_contours(E, dh)
    threshold_contours(E, dx)
    return E
    