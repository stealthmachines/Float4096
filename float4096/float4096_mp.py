# root_folder/float4096/float4096_mp.py
import mpmath
from .float4096 import prepare_prime_interpolation, fib_real, native_prime_product, P_nb

# Set precision to about 4096 bits (~1234 decimal digits)
mpmath.mp.dps = 1234

# --- Arbitrary precision Float4096 class ---

class Float4096:
    def __init__(self, value):
        if isinstance(value, Float4096):
            self.val = mpmath.mpf(value.val)
        else:
            self.val = mpmath.mpf(value)

    def __add__(self, other):
        return Float4096(self.val + self._unwrap(other))

    def __sub__(self, other):
        return Float4096(self.val - self._unwrap(other))

    def __mul__(self, other):
        return Float4096(self.val * self._unwrap(other))

    def __truediv__(self, other):
        return Float4096(self.val / self._unwrap(other))

    def __pow__(self, other):
        return Float4096(mpmath.power(self.val, self._unwrap(other)))

    def __neg__(self):
        return Float4096(-self.val)

    def __abs__(self):
        return Float4096(mpmath.fabs(self.val))

    def __float__(self):
        return float(self.val)

    def __repr__(self):
        return f"Float4096({str(self.val)})"

    def _unwrap(self, other):
        if isinstance(other, Float4096):
            return other.val
        else:
            return mpmath.mpf(other)

    def isfinite(self):
        return mpmath.isfinite(self.val)

    def isnan(self):
        return mpmath.isnan(self.val)

    def __eq__(self, other):
        return self.val == self._unwrap(other)

    def __lt__(self, other):
        return self.val < self._unwrap(other)

    def __le__(self, other):
        return self.val <= self._unwrap(other)

    def __gt__(self, other):
        return self.val > self._unwrap(other)

    def __ge__(self, other):
        return self.val >= self._unwrap(other)

# --- ComplexFloat4096 class ---

class ComplexFloat4096:
    def __init__(self, real, imag=0):
        self.real = real if isinstance(real, Float4096) else Float4096(real)
        self.imag = imag if isinstance(imag, Float4096) else Float4096(imag)

    def __add__(self, other):
        return ComplexFloat4096(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return ComplexFloat4096(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return ComplexFloat4096(r, i)

    def conjugate(self):
        return ComplexFloat4096(self.real, -self.imag)

    def abs(self):
        return Float4096(mpmath.sqrt(self.real.val**2 + self.imag.val**2))

    def __repr__(self):
        return f"ComplexFloat4096({repr(self.real)}, {repr(self.imag)})"

# --- Math functions ---

def sqrt(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.sqrt(x.val))
    else:
        return Float4096(mpmath.sqrt(mpmath.mpf(x)))

def exp(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.exp(x.val))
    else:
        return Float4096(mpmath.exp(mpmath.mpf(x)))

def log(x):
    if isinstance(x, Float4096):
        if x.val <= 0:
            raise ValueError("Log of non-positive number")
        return Float4096(mpmath.log(x.val))
    else:
        x_mpf = mpmath.mpf(x)
        if x_mpf <= 0:
            raise ValueError("Log of non-positive number")
        return Float4096(mpmath.log(x_mpf))

def log10(x):
    if isinstance(x, Float4096):
        if x.val <= 0:
            raise ValueError("Log of non-positive number")
        return Float4096(mpmath.log10(x.val))
    else:
        x_mpf = mpmath.mpf(x)
        if x_mpf <= 0:
            raise ValueError("Log of non-positive number")
        return Float4096(mpmath.log10(x_mpf))

def sin(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.sin(x.val))
    else:
        return Float4096(mpmath.sin(mpmath.mpf(x)))

def cos(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.cos(x.val))
    else:
        return Float4096(mpmath.cos(mpmath.mpf(x)))

def pow_f4096(base, exponent):
    base_val = base.val if isinstance(base, Float4096) else mpmath.mpf(base)
    exp_val = exponent.val if isinstance(exponent, Float4096) else mpmath.mpf(exponent)
    return Float4096(mpmath.power(base_val, exp_val))

# --- GRAElement class ---

class GRAElement:
    def __init__(self, n: Float4096, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None):
        if float(n) < 1:
            raise ValueError("n must be >= 1")
        self.n = Float4096(n)
        self.Omega = Float4096(Omega)
        self.base = Float4096(base)
        self.prime_interp = prime_interp or prepare_prime_interpolation()
        self._value = self._compute_r_n()

    def _compute_r_n(self) -> Float4096:
        try:
            if float(self.n) > 1000:
                return Float4096(0)
            n_int = int(float(self.n))
            Fn = fib_real(self.n)
            if Fn == Float4096(0):
                return Float4096(0)
            product = native_prime_product(n_int)
            val = Float4096((1 + mpmath.sqrt(5)) / 2) * self.Omega * Fn * pow_f4096(self.base, self.n) * product
            return sqrt(val) if val.isfinite() and val > Float4096(0) else Float4096(0)
        except Exception:
            return Float4096(0)

    @classmethod
    def from_recursive(cls, n: Float4096, prev_r_n_minus_1: 'GRAElement' = None, Omega: Float4096 = Float4096(1), base: Float4096 = Float4096(2), prime_interp=None) -> 'GRAElement':
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
        result._value = r_n if r_n.isfinite() and r_n > Float4096(0) else result._value
        return result

    def gra_multiply(self, other: 'GRAElement') -> 'GRAElement':
        if not isinstance(other, GRAElement):
            raise ValueError("GRA multiplication requires a GRAElement")
        if float(self.n) != float(other.n) + 1:
            raise ValueError("GRA multiplication requires n = m + 1")
        return GRAElement.from_recursive(self.n, prev_r_n_minus_1=other, Omega=self.Omega, base=self.base, prime_interp=self.prime_interp)

    def gra_add(self, other: 'GRAElement') -> Float4096:
        if not isinstance(other, GRAElement):
            raise ValueError("GRA addition requires a GRAElement")
        result = sqrt(self._value ** Float4096(2) + other._value ** Float4096(2))
        return result if result.isfinite() else Float4096(0)

    def __float__(self) -> float:
        return float(self._value)

    def __str__(self) -> str:
        return f"GRAElement({float(self._value)})"

# --- Fibonacci using Float4096 ---

def fibonacci(n):
    """Compute nth Fibonacci number using Binet's formula with high precision"""
    sqrt5 = sqrt(Float4096(5))
    phi = (Float4096(1) + sqrt5) / Float4096(2)
    psi = (Float4096(1) - sqrt5) / Float4096(2)
    return ((pow_f4096(phi, n) - pow_f4096(psi, n)) / sqrt5)

# --- Example usage ---

if __name__ == "__main__":
    print("4096-bit precision Float and Fibonacci Demo")
    a = Float4096("1.234567890123456789")
    b = Float4096("2.34567890123456789")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"sqrt(a) = {sqrt(a)}")
    x = GRAElement(Float4096(2))
    y = GRAElement(Float4096(1))
    print(f"x = {x}")
    print(f"x + y = {x.gra_add(y)}")
    print(f"x * y = {x.gra_multiply(y)}")
    n = 100
    fib_n = fibonacci(n)
    print(f"Fibonacci({n}) = {fib_n}")
    print(f"Integer approximation of Fibonacci({n}): {int(mpmath.nint(fib_n.val))}")
    c1 = ComplexFloat4096(a, b)
    c2 = ComplexFloat4096(b, a)
    print(f"c1 = {c1}")
    print(f"c2 = {c2}")
    print(f"c1 * c2 = {c1 * c2}")
    print(f"|c1| = {c1.abs()}")
