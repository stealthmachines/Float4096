# float4096/float4096.py
import math
from typing import List, Tuple, Union, Dict
import numpy as np
from numbers import Number
import sympy as sp
from sympy import zeta, exp as sympy_exp, I, pi, conjugate, Rational
from scipy.optimize import root_scalar
from collections import OrderedDict
import pickle
import os
import functools

# Import mpmath-based arithmetic and GRAElement from float4096_mp
from .float4096_mp import (
    Float4096,
    ComplexFloat4096,
    GRAElement,
    sqrt,
    exp,
    log,
    sin,
    cos,
    pow_f4096,
)

# Import utilities from float4096_utils
from .float4096_utils import (
    prepare_prime_interpolation,
    fib_real,
    native_prime_product,
    P_nb,
    native_cubic_spline,
    compute_spline_coefficients,
    phi,
    sqrt5,
    pi_val,
    cache_set,
    fib_cache,
    prime_cache,
    prime_product_cache,
    spline_cache,
)

# Symbolic constants
Omega = sp.Symbol("Ω", positive=True)
k = None
r = None

# Caches
zeta_cache = OrderedDict()

# FFT setup
try:
    import pyfftw
    FFT = pyfftw.interfaces.numpy_fft.fft
    IFFT = pyfftw.interfaces.numpy_fft.ifft
except ImportError:
    FFT = np.fft.fft
    IFFT = np.fft.ifft

def native_zeta(s: complex, max_terms: int = 500) -> ComplexFloat4096:
    """Approximate zeta function with mpmath precision"""
    s_key = str(s)
    if s_key in zeta_cache:
        return zeta_cache[s_key]
    real_sum = Float4096(0)
    imag_sum = Float4096(0)
    if isinstance(s, complex):
        s_real, s_imag = float(s.real), float(s.imag)
    else:
        try:
            s_complex = complex(s)
            s_real, s_imag = s_complex.real, s_complex.imag
        except (TypeError, ValueError):
            s_real, s_imag = float(s), 0.0
    for n in range(1, max_terms + 1):
        n_float = Float4096(n)
        term = Float4096(1) / pow_f4096(n_float, Float4096(s_real))
        angle = Float4096(s_imag) * log(n_float)
        real_term = term * cos(angle)
        imag_term = term * sin(angle)
        real_sum += real_term
        imag_sum += imag_term
        if max(abs(real_term), abs(imag_term)) < Float4096("1e-20"):
            break
    result = ComplexFloat4096(real_sum, imag_sum)
    cache_set(zeta_cache, s_key, result)
    return result

def solve_n_beta_for_prime(p_target: int, prime_interp, bracket=(0.1, 20)) -> Float4096:
    def objective(n_beta): return prime_interp(n_beta) - p_target
    result = root_scalar(objective, bracket=bracket, method='brentq')
    if result.converged:
        return Float4096(result.root)
    raise ValueError(f"Could not solve for n_beta corresponding to prime {p_target}")

def abs(x: Union[Float4096, 'Float4096Array', ComplexFloat4096]) -> Float4096:
    if isinstance(x, Float4096):
        return x.__abs__()
    elif isinstance(x, Float4096Array):
        return Float4096Array([abs(v) for v in x._data])
    elif isinstance(x, ComplexFloat4096):
        return x.abs()
    return Float4096(math.fabs(float(x)) if isinstance(x, (int, float)) else float(x))

def max(x: 'Float4096Array', y: Union[Float4096, Number] = None) -> Float4096:
    if y is None:
        return Float4096(max(float(v) for v in x._data))
    return x if float(x) > float(y) else Float4096(y)

class Float4096Array:
    def __init__(self, values: Union[List, Tuple, np.ndarray, 'Float4096Array']):
        if isinstance(values, (list, tuple, np.ndarray)):
            self._data = [Float4096(v) if not isinstance(v, Float4096) else v for v in values]
        elif isinstance(values, Float4096Array):
            self._data = values._data.copy()
        else:
            raise ValueError("Invalid input for Float4096Array")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Float4096, 'Float4096Array']:
        if isinstance(idx, (int, slice)):
            return self._data[idx] if isinstance(idx, int) else Float4096Array(self._data[idx])
        raise ValueError("Invalid index")

    def __setitem__(self, idx: Union[int, slice], value: Union[Float4096, Number]):
        if isinstance(idx, (int, slice)):
            self._data[idx] = Float4096(value) if not isinstance(value, Float4096) else value
        else:
            raise ValueError("Invalid index")

    def __array__(self, dtype=None) -> np.ndarray:
        return np.array([float(x) for x in self._data], dtype=dtype if dtype else np.float64)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [np.array(x) if isinstance(x, Float4096Array) else x for x in inputs]
        inputs = [Float4096Array([x]) if isinstance(x, (Float4096, Number)) else x for x in inputs]
        inputs = [np.array([float(v) for v in x._data]) if isinstance(x, Float4096Array) else x for x in inputs]
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, np.ndarray):
            return Float4096Array([Float4096(x) for x in result])
        return Float4096(result)

    def __add__(self, other: Union[Float4096, 'Float4096Array', Number]) -> 'Float4096Array':
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x + other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x + y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __mul__(self, other: Union[Float4096, 'Float4096Array', Number]) -> 'Float4096Array':
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x * other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x * y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __str__(self) -> str:
        return f"Float4096Array({[float(x) for x in self._data]})"

def log10(x: Float4096) -> Float4096:
    return log(x) / log(Float4096(10))

def linspace(start: Float4096, stop: Float4096, num: int) -> Float4096Array:
    start, stop = Float4096(start), Float4096(stop)
    step = (stop - start) / Float4096(num - 1)
    return Float4096Array([start + Float4096(i) * step for i in range(num)])

def logspace(start: Float4096, stop: Float4096, num: int) -> Float4096Array:
    start, stop = Float4096(start), Float4096(stop)
    return Float4096Array([exp(x) for x in linspace(log(start), log(stop), num)])

def mean(x: Float4096Array) -> Float4096:
    total = Float4096(0)
    for xi in x._data:
        total += xi
    return total / Float4096(len(x))

def stddev(x: Float4096Array) -> Float4096:
    mu = mean(x)
    n = len(x)
    if n <= 1:
        return Float4096(0)
    squared_diff = Float4096(0)
    for xi in x._data:
        squared_diff += (xi - mu) ** Float4096(2)
    return sqrt(squared_diff / Float4096(n - 1))

def D(n: Float4096, beta: Float4096, r: Float4096 = Float4096(1), k: Float4096 = Float4096(1), Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), scale: Float4096 = Float4096(1), prime_interp=None) -> Float4096:
    try:
        r_n = GRAElement(n + beta, Omega=Omega, base=base, prime_interp=prime_interp)
        val = scale * r_n._value
        if not val.isfinite() or val <= Float4096(0):
            return Float4096("1e-30")
        return val * (r ** k)
    except Exception:
        return Float4096("1e-30")

def D_x(x: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return D(x, Float4096(0), Omega=Omega, base=base, prime_interp=prime_interp)

def F_x(x: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / D_x(x, Omega, base, prime_interp)

def invert_D(target: Float4096, r: Float4096 = Float4096(1), k: Float4096 = Float4096(1), Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), scale: Float4096 = Float4096(1), prime_interp=None) -> Tuple[Float4096, Float4096, Float4096, Float4096, Float4096, Float4096]:
    try:
        def objective(params):
            n, beta, dynamic_scale = params
            approx = D(Float4096(n), Float4096(beta), r, k, Omega, base, scale * Float4096(dynamic_scale), prime_interp)
            return float(abs(approx - target))
        
        from scipy.optimize import minimize
        initial_guess = [1.0, 0.5, 1.0]
        bounds = [(0.1, 1000.0), (0.0, 1.0), (0.01, 100.0)]
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            n, beta, dynamic_scale = result.x
            n, beta, dynamic_scale = Float4096(n), Float4096(beta), Float4096(dynamic_scale)
            approx = D(n, beta, r, k, Omega, base, scale * dynamic_scale, prime_interp)
            uncertainty = Float4096(result.fun)
            return n, beta, dynamic_scale, uncertainty, r, k
        return None, None, None, None, None, None
    except Exception:
        return None, None, None, None, None, None

def ds_squared(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    r_n = GRAElement(n, Omega, base, prime_interp)
    r_n_minus_1 = GRAElement(n - Float4096(1), Omega, base, prime_interp)
    return (r_n._value - r_n_minus_1._value) ** Float4096(2)

def g9(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / ds_squared(n, Omega, base, prime_interp)

def R9(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / g9(n, Omega, base, prime_interp)

def grad9_r_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    r_n = GRAElement(n, Omega, base, prime_interp)
    r_n_minus_1 = GRAElement(n - Float4096(1), Omega, base, prime_interp)
    return (r_n._value - r_n_minus_1._value) / Float4096(1)

def Gamma_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    r_n = GRAElement(n, Omega, base, prime_interp)
    return Float4096(1) / r_n._value

def D_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return D(n, Float4096(0), Omega=Omega, base=base, prime_interp=prime_interp)

def T(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / D_n(n, Omega, base, prime_interp)

def Xi_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / (n * D_n(n, Omega, base, prime_interp))

def psi_9(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return D_n(n, Omega, base, prime_interp) / n

def E_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / psi_9(n, Omega, base, prime_interp)

def edge_weight(n: Float4096, m: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    if float(n) != float(m) + 1:
        return Float4096(0)
    r_n = GRAElement(n, Omega, base, prime_interp)
    r_m = GRAElement(m, Omega, base, prime_interp)
    return r_n.gra_add(r_m)

def Coil(n: Float4096, beta: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    r_n = GRAElement(n + beta, Omega, base, prime_interp)
    return ComplexFloat4096(r_n._value, Float4096(0))

def Spin(n: Float4096, beta: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    r_n = GRAElement(n + beta, Omega, base, prime_interp)
    angle = Float4096(2) * pi_val() * (n + beta)
    return ComplexFloat4096(cos(angle) * r_n._value, sin(angle) * r_n._value)

def Splice(n: Float4096, beta: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    r_n = GRAElement(n + beta, Omega, base, prime_interp)
    r_n_minus_1 = GRAElement(n + beta - Float4096(1), Omega, base, prime_interp)
    return ComplexFloat4096(r_n._value - r_n_minus_1._value, Float4096(0))

def Reflect(n: Float4096, beta: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    r_n = GRAElement(n + beta, Omega, base, prime_interp)
    return ComplexFloat4096(-r_n._value, Float4096(0))

def Coil_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    return Coil(n, Float4096(0), Omega, base, prime_interp)

def Spin_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    return Spin(n, Float4096(0), Omega, base, prime_interp)

def Splice_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    return Splice(n, Float4096(0), Omega, base, prime_interp)

def Reflect_n(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> ComplexFloat4096:
    return Reflect(n, Float4096(0), Omega, base, prime_interp)

def recursive_time(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return T(n, Omega, base, prime_interp)

def frequency(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return Float4096(1) / recursive_time(n, Omega, base, prime_interp)

def charge(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return sqrt(D_n(n, Omega, base, prime_interp))

def field_yield(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return E_n(n, Omega, base, prime_interp)

def action(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return D_n(n, Omega, base, prime_interp) * T(n, Omega, base, prime_interp)

def energy(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return field_yield(n, Omega, base, prime_interp)

def force(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return field_yield(n, Omega, base, prime_interp) / T(n, Omega, base, prime_interp)

def voltage(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return charge(n, Omega, base, prime_interp) / T(n, Omega, base, prime_interp)

def labeled_output(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Dict[str, Float4096]:
    return {
        "D_n": D_n(n, Omega, base, prime_interp),
        "T": T(n, Omega, base, prime_interp),
        "Xi_n": Xi_n(n, Omega, base, prime_interp),
        "psi_9": psi_9(n, Omega, base, prime_interp),
        "E_n": E_n(n, Omega, base, prime_interp),
        "charge": charge(n, Omega, base, prime_interp),
        "field_yield": field_yield(n, Omega, base, prime_interp),
        "action": action(n, Omega, base, prime_interp),
        "energy": energy(n, Omega, base, prime_interp),
        "force": force(n, Omega, base, prime_interp),
        "voltage": voltage(n, Omega, base, prime_interp),
    }

def field_automorphisms(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> List[ComplexFloat4096]:
    return [
        Coil_n(n, Omega, base, prime_interp),
        Spin_n(n, Omega, base, prime_interp),
        Splice_n(n, Omega, base, prime_interp),
        Reflect_n(n, Omega, base, prime_interp),
    ]

def field_tension(n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> Float4096:
    return grad9_r_n(n, Omega, base, prime_interp)

class GoldenClassField:
    def __init__(self, s: List[Rational], x: Union[List[Float4096], Float4096Array], Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None):
        self.s = s if isinstance(s, list) else [s]
        self.x = Float4096Array(x) if not isinstance(x, Float4096Array) else x
        self.Omega = Float4096(Omega)
        self.base = Float4096(base)
        self.prime_interp = prime_interp or prepare_prime_interpolation()
        self.field_dict = {}
        self._compute_field()

    def _compute_field(self):
        for s in self.s:
            for x in self.x:
                n_beta = Float4096(s) + x
                r_n = GRAElement(n_beta, self.Omega, self.base, self.prime_interp)
                self.field_dict[(s, x)] = ComplexFloat4096(r_n._value, Float4096(0))

    def as_dict(self) -> Dict[Tuple[Rational, Float4096], ComplexFloat4096]:
        return self.field_dict
