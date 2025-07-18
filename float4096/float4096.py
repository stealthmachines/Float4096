import math
from typing import List, Tuple, Union, Dict
import numpy as np
from numbers import Number
import sympy as sp
from sympy import zeta, exp as sympy_exp, I, pi, conjugate, Rational
from scipy.optimize import root_scalar
from collections import OrderedDict

# Constants
BASE = 4096
DIGITS_PER_NUMBER = 64  # ~768 bits

def generate_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]

PRIMES = generate_primes(104729)[:10000]  # First 10,000 primes
phi = None  # Initialized later
sqrt5 = None
Omega = sp.Symbol("Ω", positive=True)
k = None  # Initialized later
r = None
pi_val = None

# Caches
MAX_CACHE_SIZE = 10000
fib_cache = OrderedDict()
prime_cache = OrderedDict()
zeta_cache = OrderedDict()
prime_product_cache = OrderedDict()
spline_cache = OrderedDict()

def cache_set(cache, key, value):
    cache[key] = value
    if len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)

# FFT setup
try:
    import pyfftw
    FFT = pyfftw.interfaces.numpy_fft.fft
    IFFT = pyfftw.interfaces.numpy_fft.ifft
except ImportError:
    FFT = np.fft.fft
    IFFT = np.fft.ifft

def native_prime_product(n: int) -> 'Float4096':
    if n < 0 or n > len(PRIMES):
        raise ValueError(f"n must be between 0 and {len(PRIMES)}")
    if n in prime_product_cache:
        return prime_product_cache[n]
    product = Float4096(1)
    for p in PRIMES[:n]:
        product *= Float4096(p)
    cache_set(prime_product_cache, n, product)
    return product

def prepare_prime_interpolation(primes_list=PRIMES):
    indices = [float(i + 1) for i in range(len(primes_list))]
    recursive_index_phi = [float(log(Float4096(i + 1)) / log(phi)) for i in indices]
    return lambda x: native_cubic_spline(x, recursive_index_phi, primes_list)

def fib_real(n: 'Float4096') -> 'Float4096':
    """Native base4096 Fibonacci using Binet's formula"""
    n_float = float(n)
    if n_float > 70:
        return Float4096(0)
    if n_float in fib_cache:
        return fib_cache[n_float]
    try:
        term1 = pow_f4096(phi, n) / sqrt5
        term2 = pow_f4096(Float4096(1) / phi, n) * cos(pi_val * n)
        result = term1 - term2
        if not result.is_finite():
            result = Float4096(0)
        cache_set(fib_cache, n_float, result)
        return result
    except Exception:
        return Float4096(0)

def native_zeta(s: complex, max_terms: int = 500) -> 'ComplexFloat4096':
    """Approximate zeta function in base4096"""
    s_key = str(s)
    if s_key in zeta_cache:
        return zeta_cache[s_key]
    real_sum = Float4096(0)
    imag_sum = Float4096(0)
    s_real, s_imag = float(s.real), float(s.imag)
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

def compute_spline_coefficients(x_points: List[float], y_points: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute cubic spline coefficients (a, b, c, d) for each segment"""
    n = len(x_points) - 1
    h = [x_points[i + 1] - x_points[i] for i in range(n)]
    a = y_points[:-1]
    c = [0.0] * (n + 1)
    alpha = [0.0] * n
    for i in range(1, n):
        alpha[i] = 3 * ((y_points[i + 1] - y_points[i]) / h[i] - (y_points[i] - y_points[i - 1]) / h[i - 1])
    l = [1.0] * (n + 1)
    mu = [0.0] * n
    z = [0.0] * (n + 1)
    for i in range(1, n):
        l[i] = 2 * (x_points[i + 1] - x_points[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    b = [0.0] * n
    d = [0.0] * n
    for i in range(n - 1, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (y_points[i + 1] - y_points[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return a, b, c, d

def native_cubic_spline(x: float, x_points: List[float], y_points: List[float]) -> float:
    """Full cubic spline interpolation in base4096"""
    x_key = (x, tuple(x_points))
    if x_key in spline_cache:
        return float(spline_cache[x_key])
    n = len(x_points) - 1
    a, b, c, d = compute_spline_coefficients(x_points, y_points)
    for i in range(n):
        if x_points[i] <= x <= x_points[i + 1]:
            t = x - x_points[i]
            result = a[i] + b[i] * t + c[i] * t**2 + d[i] * t**3
            cache_set(spline_cache, x_key, Float4096(result))
            return result
    result = y_points[-1] if x > x_points[-1] else y_points[0]
    cache_set(spline_cache, x_key, Float4096(result))
    return result

def P_nb(n_beta: 'Float4096', prime_interp) -> 'Float4096':
    n_float = float(n_beta)
    if n_float in prime_cache:
        return prime_cache[n_float]
    result = Float4096(prime_interp(n_float))
    cache_set(prime_cache, n_float, result)
    return result

def solve_n_beta_for_prime(p_target: int, prime_interp, bracket=(0.1, 20)) -> 'Float4096':
    def objective(n_beta): return prime_interp(n_beta) - p_target
    result = root_scalar(objective, bracket=bracket, method='brentq')
    if result.converged:
        return Float4096(result.root)
    raise ValueError(f"Could not solve for n_beta corresponding to prime {p_target}")

def abs(x: Union['Float4096', 'Float4096Array', 'ComplexFloat4096']) -> 'Float4096':
    if isinstance(x, Float4096):
        return x.__absrect()
    elif isinstance(x, Float4096Array):
        return Float4096Array([abs(v) for v in x._data])
    elif isinstance(x, ComplexFloat4096):
        return x.abs()
    return Float4096(math.fabs(float(x)) if isinstance(x, (int, float)) else float(x))

def max(x: 'Float4096Array', y: Union['Float4096', Number] = None) -> 'Float4096':
    if y is None:
        return Float4096(max(float(v) for v in x._data))
    return x if float(x) > float(y) else Float4096(y)

def pow_f4096(x: 'Float4096', y: 'Float4096') -> 'Float4096':
    return Float4096(math.pow(float(x), float(y)))

class Float4096:
    def __init__(self, value: Union[float, int, str, List[int], 'Float4096'] = 0, exponent: int = 0, sign: int = 1):
        self.sign = 1 if (isinstance(value, (int, float)) and value >= 0) else -1 if isinstance(value, (int, float)) else sign
        self.exponent = exponent
        if isinstance(value, (int, float)):
            self.digits = self._from_float(float(value))
        elif isinstance(value, str):
            try:
                # Try to parse the string as a float first (e.g., for "1e-20")
                value_float = float(value)
                self.digits = self._from_float(value_float)
            except ValueError:
                # If not a valid float, assume it's a base4096 encoded string
                self.digits = self._from_base4096_str(value)
        elif isinstance(value, list):
            self.digits = value[:DIGITS_PER_NUMBER]
            self.digits += [0] * (DIGITS_PER_NUMBER - len(self.digits))
        elif isinstance(value, Float4096):
            self.digits = value.digits[:]
            self.exponent = value.exponent
            self.sign = value.sign
        else:
            raise ValueError("Invalid input for Float4096")
        self.normalize()

    def _from_float(self, value: float) -> List[int]:
        if value == 0:
            return [0] * DIGITS_PER_NUMBER
        value = math.fabs(value)
        exponent = int(math.floor(math.log(value, BASE))) if value != 0 else 0
        self.exponent = exponent
        mantissa = value / (BASE ** exponent)
        digits = []
        for _ in range(DIGITS_PER_NUMBER):
            digit = int(mantissa * BASE)
            digits.append(digit)
            mantissa = (mantissa * BASE) - digit
        return digits

    def _from_base4096_str(self, b4096_str: str) -> List[int]:
        import base4096
        binary = base4096.decode(b4096_str)
        value = np.frombuffer(binary, dtype=np.float64)[0]
        return self._from_float(value)

    def to_float(self) -> float:
        result = 0.0
        for i, digit in enumerate(self.digits):
            result += digit * (BASE ** (self.exponent - i))
        return self.sign * result

    def normalize(self):
        while self.digits and self.digits[0] == 0:
            self.digits.pop(0)
            self.exponent -= 1
        self.digits = self.digits[:DIGITS_PER_NUMBER] + [0] * (DIGITS_PER_NUMBER - len(self.digits))
        if all(d == 0 for d in self.digits):
            self.sign = 1
            self.exponent = 0

    def is_finite(self) -> bool:
        return all(0 <= d < BASE for d in self.digits) and abs(self.exponent) < 1000000

    def __add__(self, other: 'Float4096') -> 'Float4096':
        if not isinstance(other, Float4096):
            other = Float4096(other)
        if self.exponent < other.exponent:
            return other + self
        result_digits = [0] * DIGITS_PER_NUMBER
        carry = 0
        offset = self.exponent - other.exponent
        for i in range(DIGITS_PER_NUMBER - 1, -1, -1):
            other_digit = other.digits[i - offset] if 0 <= i - offset < DIGITS_PER_NUMBER else 0
            total = self.digits[i] + other_digit + carry
            result_digits[i] = total % BASE
            carry = total // BASE
        result = Float4096(0)
        result.digits = result_digits
        result.exponent = self.exponent
        result.sign = self.sign if abs(self) >= abs(other) else other.sign
        result.normalize()
        return result

    def __sub__(self, other: 'Float4096') -> 'Float4096':
        if not isinstance(other, Float4096):
            other = Float4096(other)
        other_neg = Float4096(other.digits, other.exponent, -other.sign)
        return self + other_neg

    def fft_multiply(self, other: 'Float4096') -> 'Float4096':
        if not isinstance(other, Float4096):
            other = Float4096(other)
        n = 2 * DIGITS_PER_NUMBER
        a = np.array(self.digits + [0] * DIGITS_PER_NUMBER, dtype=np.float64)
        b = np.array(other.digits + [0] * DIGITS_PER_NUMBER, dtype=np.float64)
        fa = FFT(a, n)
        fb = FFT(b, n)
        fc = fa * fb
        c = IFFT(fc).real
        result_digits = [0] * (2 * DIGITS_PER_NUMBER)
        carry = 0
        for i in range(2 * DIGITS_PER_NUMBER - 1, -1, -1):
            total = int(c[i] + carry + 0.5)
            result_digits[i] = total % BASE
            carry = total // BASE
        result = Float4096(0)
        result.digits = result_digits[:DIGITS_PER_NUMBER]
        result.exponent = self.exponent + other.exponent
        result.sign = self.sign * other.sign
        result.normalize()
        return result

    def __mul__(self, other: 'Float4096') -> 'Float4096':
        return self.fft_multiply(other)

    def __truediv__(self, other: 'Float4096') -> 'Float4096':
        if not isinstance(other, Float4096):
            other = Float4096(other)
        if float(other) < Float4096("1e-30"):
            raise ValueError("Division by near-zero value")
        result_digits = [0] * DIGITS_PER_NUMBER
        remainder = Float4096(self.digits, self.exponent, self.sign)
        divisor = Float4096(other.digits, other.exponent, other.sign)
        for i in range(DIGITS_PER_NUMBER):
            quotient_digit = 0
            while remainder >= divisor:
                remainder = remainder - divisor
                quotient_digit += 1
            result_digits[i] = quotient_digit
            remainder = remainder * Float4096(BASE)
        result = Float4096(result_digits, self.exponent - other.exponent, self.sign * other.sign)
        result.normalize()
        return result

    def __pow__(self, other: 'Float4096') -> 'Float4096':
        n = float(other)
        if n == int(n):
            result = Float4096(1)
            for _ in range(int(n)):
                result = result.fft_multiply(self)
            return result
        return pow_f4096(self, other)

    def __eq__(self, other: 'Float4096') -> bool:
        if not isinstance(other, Float4096):
            other = Float4096(other)
        return self.sign == other.sign and self.exponent == other.exponent and self.digits == other.digits

    def __lt__(self, other: 'Float4096') -> bool:
        if not isinstance(other, Float4096):
            other = Float4096(other)
        if self.sign != other.sign:
            return self.sign < other.sign
        if self.exponent != other.exponent:
            return (self.exponent < other.exponent) if self.sign > 0 else (self.exponent > other.exponent)
        return self.digits < other.digits

    def __le__(self, other: 'Float4096') -> bool:
        return self == other or self < other

    def __gt__(self, other: 'Float4096') -> bool:
        return not self <= other

    def __ge__(self, other: 'Float4096') -> bool:
        return not self < other

    def __abs__(self) -> 'Float4096':
        return Float4096(self.digits, self.exponent, 1)

    def __float__(self) -> float:
        return self.to_float()

    def __str__(self) -> str:
        return f"Float4096({self.to_float()})"

    def __repr__(self) -> str:
        return f"Float4096(digits={self.digits[:5]}..., exponent={self.exponent}, sign={self.sign})"

    def to_base4096(self) -> str:
        import base4096
        return base4096.encode(self.digits_to_bytes())

    def digits_to_bytes(self) -> bytes:
        return b''.join(d.to_bytes(2, 'big') for d in self.digits)

class ComplexFloat4096:
    def __init__(self, real: 'Float4096', imag: 'Float4096' = Float4096(0)):
        self.real = Float4096(real)
        self.imag = Float4096(imag)

    def __add__(self, other: 'ComplexFloat4096') -> 'ComplexFloat4096':
        if not isinstance(other, ComplexFloat4096):
            other = ComplexFloat4096(Float4096(other))
        return ComplexFloat4096(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other: 'ComplexFloat4096') -> 'ComplexFloat4096':
        if not isinstance(other, ComplexFloat4096):
            other = ComplexFloat4096(Float4096(other))
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexFloat4096(real, imag)

    def __truediv__(self, other: 'ComplexFloat4096') -> 'ComplexFloat4096':
        if not isinstance(other, ComplexFloat4096):
            other = ComplexFloat4096(Float4096(other))
        denom = other.real * other.real + other.imag * other.imag
        if float(denom) < Float4096("1e-30"):
            raise ValueError("Division by near-zero complex magnitude")
        real = (self.real * other.real + self.imag * other.imag) / denom
        imag = (self.imag * other.real - self.real * other.imag) / denom
        return ComplexFloat4096(real, imag)

    def conjugate(self) -> 'ComplexFloat4096':
        return ComplexFloat4096(self.real, -self.imag)

    def abs(self) -> 'Float4096':
        return sqrt(self.real * self.real + self.imag * self.imag)

    def exp(self) -> 'ComplexFloat4096':
        e_real = exp(self.real)
        return ComplexFloat4096(e_real * cos(self.imag), e_real * sin(self.imag))

    def __float__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __str__(self) -> str:
        return f"ComplexFloat4096({float(self.real)} + {float(self.imag)}i)"

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

    def __getitem__(self, idx: Union[int, slice]) -> Union['Float4096', 'Float4096Array']:
        if isinstance(idx, (int, slice)):
            return self._data[idx] if isinstance(idx, int) else Float4096Array(self._data[idx])
        raise ValueError("Invalid index")

    def __setitem__(self, idx: Union[int, slice], value: Union['Float4096', Number]):
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

    def __add__(self, other: Union['Float4096', 'Float4096Array', Number]) -> 'Float4096Array':
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x + other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x + y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __mul__(self, other: Union['Float4096', 'Float4096Array', Number]) -> 'Float4096Array':
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x * other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x * y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __str__(self) -> str:
        return f"Float4096Array({[float(x) for x in self._data]})"

class GRAElement:
    def __init__(self, n: 'Float4096', Omega: 'Float4096' = Float4096(1), base: 'Float4096' = Float4096(2), prime_interp=None):
        if float(n) < 1:
            raise ValueError("n must be >= 1")
        self.n = Float4096(n)
        self.Omega = Float4096(Omega)
        self.base = Float4096(base)
        self.prime_interp = prime_interp or prepare_prime_interpolation()
        self._value = self._compute_r_n()

    def _compute_r_n(self) -> 'Float4096':
        try:
            if float(self.n) > 1000:
                return Float4096(0)
            n_int = int(float(self.n))
            Fn = fib_real(self.n)
            if Fn == Float4096(0):
                return Float4096(0)
            product = native_prime_product(n_int)
            val = phi * self.Omega * Fn * pow(self.base, self.n) * product
            return sqrt(val) if val.is_finite() and val > Float4096(0) else Float4096(0)
        except Exception:
            return Float4096(0)

    @classmethod
    def from_recursive(cls, n: 'Float4096', prev_r_n_minus_1: 'GRAElement' = None, Omega: 'Float4096' = Float4096(1), base: 'Float4096' = Float4096(2), prime_interp=None) -> 'GRAElement':
        n = Float4096(n)
        if float(n) < 1:
            raise ValueError("n must be >= 1")
        if float(n) == 1:
            return cls(n, Omega, base, prime_interp)
        if prev_r_n_minus_1 is None:
            prev_r_n_minus_1 = cls(n - Float4096(1), Omega, base, prime_interp)
        Fn = fib_real(n)
        Fn_minus_1 = fib_real(n - Float4096(1))
        if Fn == Float4096(0) or Fn_minus_1 == Float4096(0):
            return cls(n, Omega, base, prime_interp)
        p_n = P_nb(n, prime_interp or prepare_prime_interpolation())
        factor = sqrt(Float4096(2) * p_n * (Fn / Fn_minus_1))
        r_n = prev_r_n_minus_1._value * factor
        result = cls(n, Omega, base, prime_interp)
        result._value = r_n if r_n.is_finite() and r_n > Float4096(0) else result._value
        return result

    def gra_multiply(self, other: 'GRAElement') -> 'GRAElement':
        if not isinstance(other, GRAElement):
            raise ValueError("GRA multiplication requires a GRAElement")
        if float(self.n) != float(other.n) + 1:
            raise ValueError("GRA multiplication requires n = m + 1")
        return GRAElement.from_recursive(self.n, prev_r_n_minus_1=other, Omega=self.Omega, base=self.base, prime_interp=self.prime_interp)

    def gra_add(self, other: 'GRAElement') -> 'Float4096':
        if not isinstance(other, GRAElement):
            raise ValueError("GRA addition requires a GRAElement")
        result = sqrt(self._value ** Float4096(2) + other._value ** Float4096(2))
        return result if result.is_finite() else Float4096(0)

    def __float__(self) -> float:
        return float(self._value)

    def __str__(self) -> str:
        return f"GRAElement({float(self._value)})"

def sqrt(x: 'Float4096') -> 'Float4096':
    if isinstance(x, Float4096Array):
        return Float4096Array([sqrt(v) for v in x._data])
    if float(x) < 0:
        raise ValueError("Square root of negative number")
    if float(x) == 0:
        return Float4096(0)
    guess = Float4096(float(x) ** 0.5)
    for _ in range(10):
        guess = (guess + (x / guess)) / Float4096(2)
    return guess

def exp(x: 'Float4096') -> 'Float4096':
    if isinstance(x, Float4096Array):
        return Float4096Array([exp(v) for v in x._data])
    x_float = float(x)
    if abs(x_float) < 1e-10:
        return Float4096(1)
    result = Float4096(1)
    term = Float4096(1)
    for n in range(1, 50):
        term *= x / Float4096(n)
        result += term
        if abs(term) < Float4096("1e-20"):
            break
    return result

def log(x: 'Float4096') -> 'Float4096':
    if isinstance(x, Float4096Array):
        return Float4096Array([log(v) for v in x._data])
    if float(x) <= 0:
        raise ValueError("Log of non-positive number")
    x_float = float(x)
    if abs(x_float - 1) < 1e-10:
        return Float4096(0)
    return Float4096(math.log(x_float))

def sin(x: 'Float4096') -> 'Float4096':
    if isinstance(x, Float4096Array):
        return Float4096Array([sin(v) for v in x._data])
    result = x
    term = x
    x_sq = x * x
    for n in range(1, 50):
        term *= -x_sq / Float4096(2 * n * (2 * n + 1))
        result += term
        if abs(term) < Float4096("1e-20"):
            break
    return result

def cos(x: 'Float4096') -> 'Float4096':
    if isinstance(x, Float4096Array):
        return Float4096Array([cos(v) for v in x._data])
    result = Float4096(1)
    term = Float4096(1)
    x_sq = x * x
    for n in range(1, 50):
        term *= -x_sq / Float4096(2 * n * (2 * n - 1))
        result += term
        if abs(term) < Float4096("1e-20"):
            break
    return result

def pi_val() -> 'Float4096':
    return Float4096(math.pi)

def log10(x: 'Float4096') -> 'Float4096':
    return log(x) / log(Float4096(10))

def linspace(start: 'Float4096', stop: 'Float4096', num: int) -> 'Float4096Array':
    start, stop = Float4096(start), Float4096(stop)
    step = (stop - start) / Float4096(num - 1)
    return Float4096Array([start + Float4096(i) * step for i in range(num)])

def logspace(start: 'Float4096', stop: 'Float4096', num: int) -> 'Float4096Array':
    start, stop = Float4096(start), Float4096(stop)
    return Float4096Array([exp(x) for x in linspace(log(start), log(stop), num)])

def mean(x: 'Float4096Array') -> 'Float4096':
    total = Float4096(0)
    for xi in x._data:
        total += xi
    return total / Float4096(len(x))

def stddev(x: 'Float4096Array') -> 'Float4096':
    mu = mean(x)
    n = len(x)
    if n <= 1:
        return Float4096(0)
    squared_diff = Float4096(0)
    for xi in x._data:
        squared_diff += (xi - mu) ** Float4096(2)
    return sqrt(squared_diff / Float4096(n - 1))

def D(n: 'Float4096', beta: 'Float4096', r: 'Float4096' = Float4096(1), k: 'Float4096' = Float4096(1), Omega: 'Float4096' = Float4096(1), base: 'Float4096' = Float4096(2), scale: 'Float4096' = Float4096(1), prime_interp=None) -> 'Float4096':
    try:
        r_n = GRAElement(n + beta, Omega=Omega, base=base, prime_interp=prime_interp)
        val = scale * r_n._value
        if not val.is_finite() or val <= Float4096(0):
            return Float4096("1e-30")
        return val * (r ** k)
    except Exception:
        return Float4096("1e-30")

def D_x(x_val: 'Float4096', s, prime_interp) -> 'ComplexFloat4096':
    P = P_nb(x_val, prime_interp)
    F = fib_real(x_val)
    zeta_val = native_zeta(s)
    product = phi * F * pow(Float4096(2), x_val) * P * Omega
    s_k = ComplexFloat4096(Float4096(float(s.real ** k)), Float4096(float(s.imag ** k)))
    return sqrt(product) * s_k

def F_x(x_val: 'Float4096', s, prime_interp) -> 'ComplexFloat4096':
    pi_x = ComplexFloat4096(cos(pi_val * x_val), sin(pi_val * x_val)) * native_zeta(s)
    return D_x(x_val, s, prime_interp) * pi_x

def invert_D(value: 'Float4096', r: 'Float4096' = Float4096(1), k: 'Float4096' = Float4096(1), Omega: 'Float4096' = Float4096(1), base: 'Float4096' = Float4096(2), scale: 'Float4096' = Float4096(1), max_n: 'Float4096' = Float4096(5000), steps: int = 1000, prime_interp=None) -> Tuple:
    candidates = []
    try:
        value_abs = abs(value)
        value_safe = value_abs if value_abs > Float4096("1e-30") else Float4096("1e-30")
        log_val = log10(value_safe)
        start_sf = max(log_val - Float4096(3), Float4096(-10))
        end_sf = min(log_val + Float4096(3), Float4096(10))
        scale_factors = pow_f4096(Float4096(10), linspace(start_sf, end_sf, 10))
        max_n_val = min(5000, max(1000, int(500 * float(abs(log_val)))))
        steps_val = min(2000, max(500, int(200 * float(abs(log_val)))))
        n_values = logspace(Float4096(0), log10(Float4096(max_n_val)), steps_val) if float(log_val) > 2 else linspace(Float4096(0), Float4096(max_n_val), steps_val)
        r_values = Float4096Array([0.5, 1.0, 2.0])
        k_values = Float4096Array([0.5, 1.0, 2.0])
        for n in n_values:
            for beta in linspace(Float4096(0), Float4096(1), 5):
                for dynamic_scale in scale_factors:
                    for r_val in r_values:
                        for k_val in k_values:
                            val = D(n, beta, r_val, k_val, Omega, base, scale * dynamic_scale, prime_interp)
                            if val.is_finite():
                                diff = abs(val - value_abs)
                                candidates.append((diff, n, beta, dynamic_scale, r_val, k_val))
        if not candidates:
            return None, None, None, None, None, None
        candidates.sort(key=lambda x: float(x[0]))
        diff, n, beta, dynamic_scale, r_val, k_val = candidates[0]
        return n, beta, dynamic_scale, diff * Float4096("0.01"), r_val, k_val
    except Exception:
        return None, None, None, None, None, None

def ds_squared(n: 'Float4096', Omega: 'Float4096' = Float4096(1), base: 'Float4096' = Float4096(2), prime_interp=None) -> 'Float4096':
    Fn = fib_real(n)
    n_int = int(float(n))
    prime_interp = prime_interp or prepare_prime_interpolation()
    product = native_prime_product(n_int)
    term = Fn * pow(base, n) * product
    return phi * Omega * term

def g9(n: 'Float4096', prime_interp=None) -> 'Float4096':
    r_n = GRAElement(n, prime_interp=prime_interp)._value
    r_n_plus = GRAElement(n + Float4096(1e-5), prime_interp=prime_interp)._value
    r_n_minus = GRAElement(n - Float4096(1e-5), prime_interp=prime_interp)._value
    h = Float4096(1e-5)
    return (r_n_plus ** 2 - 2 * r_n ** 2 + r_n_minus ** 2) / (h ** 2)

def R9(n: 'Float4096', prime_interp=None) -> 'Float4096':
    r_n = GRAElement(n, prime_interp=prime_interp)._value
    r_n_plus = GRAElement(n + Float4096(1e-5), prime_interp=prime_interp)._value
    r_n_minus = GRAElement(n - Float4096(1e-5), prime_interp=prime_interp)._value
    h = Float4096(1e-5)
    return (r_n_plus - 2 * r_n + r_n_minus) / (h ** 2)

def grad9_r_n(n: 'Float4096', prime_interp=None) -> 'Float4096':
    r_n_minus_1 = GRAElement(n - Float4096(1), prime_interp=prime_interp)._value
    Fn = fib_real(n)
    Fn_minus_1 = fib_real(n - Float4096(1))
    prime_interp = prime_interp or prepare_prime_interpolation()
    p_n = P_nb(n, prime_interp)
    factor = sqrt(Float4096(2) * p_n * (Fn / Fn_minus_1)) - Float4096(1)
    return r_n_minus_1 * factor

def Gamma_n(n: 'Float4096', prime_interp=None) -> 'Float4096':
    r_n = GRAElement(n, prime_interp=prime_interp)._value
    r_n_plus = GRAElement(n + Float4096(1e-5), prime_interp=prime_interp)._value
    h = Float4096(1e-5)
    return Float4096(0.5) * (log(r_n_plus) - log(r_n)) / h

def D_n(n: 'Float4096', prime_interp=None) -> 'Float4096':
    r_n = GRAElement(n, prime_interp=prime_interp)._value
    r_n_plus = GRAElement(n + Float4096(1e-5), prime_interp=prime_interp)._value
    h = Float4096(1e-5)
    partial = (r_n_plus - r_n) / h
    return partial + Gamma_n(n, prime_interp)

def T(X_n: Tuple['Float4096', 'Float4096', 'Float4096', 'Float4096'], delta_n: 'Float4096' = Float4096(1), prime_interp=None) -> Tuple['Float4096', 'Float4096', 'Float4096', 'Float4096']:
    n, Fn, p_n, r_n = X_n
    n_new = n + delta_n
    Fn_new = fib_real(n_new)
    prime_interp = prime_interp or prepare_prime_interpolation()
    p_n_new = P_nb(n_new, prime_interp)
    r_n_new = GRAElement(n_new, prime_interp=prime_interp)._value
    return (n_new, Fn_new, p_n_new, r_n_new)

def Xi_n(n: 'Float4096', prime_interp=None) -> Tuple['Float4096', 'Float4096']:
    r_n = GRAElement(n, prime_interp=prime_interp)._value
    Fn = fib_real(n)
    Fn_minus_1 = fib_real(n - Float4096(1))
    prime_interp = prime_interp or prepare_prime_interpolation()
    p_n = P_nb(n, prime_interp)
    omega_n = sqrt(Float4096(2) * p_n * (Fn / Fn_minus_1))
    return (r_n, omega_n)

def psi_9(n: 'Float4096', tau_n: 'Float4096' = Float4096(1), A_n: 'Float4096' = Float4096(1), prime_interp=None) -> 'ComplexFloat4096':
    _, omega_n = Xi_n(n, prime_interp)
    angle = omega_n * tau_n
    return ComplexFloat4096(A_n * cos(angle), A_n * sin(angle))

def E_n(n: 'Float4096', tau_n: 'Float4096' = Float4096(1), prime_interp=None) -> Tuple['Float4096', 'Float4096']:
    r_n = GRAElement(n, prime_interp=prime_interp)._value
    return (r_n, tau_n)

def edge_weight(n: 'Float4096', prime_interp=None) -> 'Float4096':
    Fn = fib_real(n)
    Fn_minus_1 = fib_real(n - Float4096(1))
    prime_interp = prime_interp or prepare_prime_interpolation()
    p_n = P_nb(n, prime_interp)
    return sqrt(Float4096(2) * p_n * (Fn / Fn_minus_1))

def Coil(f: 'Float4096') -> 'ComplexFloat4096':
    angle = pi_val * f
    return ComplexFloat4096(cos(angle), sin(angle))

def Spin(f: 'Float4096', s, prime_interp) -> 'ComplexFloat4096':
    return F_x(f, s, prime_interp).conjugate()

def Splice(f: 'Float4096', s, prime_interp) -> 'ComplexFloat4096':
    s_conj = 1 - s
    return F_x(f, s_conj, prime_interp)

def Reflect(f: 'Float4096', s, prime_interp) -> 'ComplexFloat4096':
    return F_x(-f, s, prime_interp)

def apply_operator(op, x: 'Float4096', n: int, s=None, prime_interp=None) -> 'ComplexFloat4096':
    result = ComplexFloat4096(x)
    for _ in range(n):
        result = op(result.real, s, prime_interp) if op in (Spin, Splice, Reflect) else op(result.real)
    return result

def M_O(n: 'Float4096', x: 'Float4096', O, s=None, prime_interp=None) -> 'ComplexFloat4096':
    return apply_operator(O, x, int(float(n)), s, prime_interp)

def Coil_n(x: 'Float4096', n: int) -> 'ComplexFloat4096':
    return apply_operator(Coil, x, n)

def Spin_n(x: 'Float4096', n: int, s, prime_interp) -> 'ComplexFloat4096':
    return apply_operator(Spin, x, n, s, prime_interp)

def Splice_n(x: 'Float4096', n: int, s, prime_interp) -> 'ComplexFloat4096':
    return apply_operator(Splice, x, n, s, prime_interp)

def Reflect_n(x: 'Float4096', n: int, s, prime_interp) -> 'ComplexFloat4096':
    return apply_operator(Reflect, x, n, s, prime_interp)

def recursive_time(n: 'Float4096') -> 'Float4096':
    return pow_f4096(phi, -n)

def frequency(n: 'Float4096') -> 'Float4096':
    return Float4096(1) / recursive_time(n)

def charge(n: 'Float4096') -> 'Float4096':
    return pow_f4096(recursive_time(n), 3)

def field_yield(n: 'Float4096', m_val: 'Float4096') -> 'Float4096':
    return pow_f4096(m_val, 2) / pow_f4096(recursive_time(n), 7)

def action(n: 'Float4096', m_val: 'Float4096') -> 'Float4096':
    return field_yield(n, m_val) * pow_f4096(charge(n), 2)

def energy(n: 'Float4096', m_val: 'Float4096') -> 'Float4096':
    return action(n, m_val) * frequency(n)

def force(n: 'Float4096', m_val: 'Float4096') -> 'Float4096':
    return energy(n, m_val) / m_val

def voltage(n: 'Float4096', m_val: 'Float4096') -> 'Float4096':
    return energy(n, m_val) / charge(n)

def labeled_output(n: 'Float4096', m_val: 'Float4096') -> Dict[str, 'Float4096']:
    return {
        "Hz": frequency(n),
        "Time s": recursive_time(n),
        "Charge C": charge(n),
        "Yield Ω": field_yield(n, m_val),
        "Action h": action(n, m_val),
        "Energy E": energy(n, m_val),
        "Force F": force(n, m_val),
        "Voltage V": voltage(n, m_val),
    }

class GoldenClassField:
    def __init__(self, s_list: List, x_list: List['Float4096'], prime_interp):
        self.s_list = [sp.sympify(s) for s in s_list]
        self.x_list = [Float4096(x) for x in x_list]
        self.prime_interp = prime_interp
        self.field_generators = []
        self.field_names = []
        self.field_cache = {}
        self.construct_class_field()

    def construct_class_field(self):
        for s in self.s_list:
            for x in self.x_list:
                key = (str(s), float(x))
                if key not in self.field_cache:
                    self.field_cache[key] = F_x(x, s, self.prime_interp)
                self.field_generators.append(self.field_cache[key])
                self.field_names.append(f"F_{float(x):.4f}_s_{s}")

    def as_dict(self) -> Dict[str, 'ComplexFloat4096']:
        return dict(zip(self.field_names, self.field_generators))

    def display(self):
        for name, val in self.as_dict().items():
            print(f"{name} = {float(val)}")

    def reciprocity_check(self):
        print("\nReciprocity Tests: F_x(s) * F_x(1-s)")
        for s in self.s_list:
            for x in self.x_list:
                try:
                    s_conj = 1 - s
                    key1 = (str(s), float(x))
                    key2 = (str(s_conj), float(x))
                    if key1 not in self.field_cache:
                        self.field_cache[key1] = F_x(x, s, self.prime_interp)
                    if key2 not in self.field_cache:
                        self.field_cache[key2] = F_x(x, s_conj, self.prime_interp)
                    prod = self.field_cache[key1] * self.field_cache[key2]
                    print(f"x={float(x):.4f}, s={s}, F_x(s)·F_x(1-s) = {float(prod)}")
                except Exception as e:
                    print(f"Failed for x={float(x)}, s={s}: {e}")

def field_automorphisms(F_val: 'ComplexFloat4096', x_val: 'Float4096', s, prime_interp) -> Dict[str, 'ComplexFloat4096']:
    s = sp.sympify(s)
    return {
        "F_x(s)": F_val,
        "F_x(1-s)": Splice(x_val, s, prime_interp),
        "F_-x(s)": Reflect(x_val, s, prime_interp),
        "conjugate(F)": F_val.conjugate(),
    }

def field_tension(F_val: 'ComplexFloat4096', C_val, m_val: 'Float4096', s_val) -> 'Float4096':
    return Float4096(float((F_val.abs() * m_val * Float4096(float(s_val))) / (C_val**2)))

# Initialize constants
phi = Float4096((1 + math.sqrt(5)) / 2)
sqrt5 = Float4096(math.sqrt(5))
pi_val = Float4096(math.pi)
k = Float4096(-1)
r = Float4096(1)
