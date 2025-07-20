import mpmath

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
        return Float4096(mpmath.log(x.val))
    else:
        return Float4096(mpmath.log(mpmath.mpf(x)))

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
    def __init__(self, value):
        # value can be int, float, string, or Float4096
        self.value = Float4096(value)

    def __add__(self, other):
        return GRAElement(self.value + other.value)

    def __sub__(self, other):
        return GRAElement(self.value - other.value)

    def __mul__(self, other):
        return GRAElement(self.value * other.value)

    def __truediv__(self, other):
        return GRAElement(self.value / other.value)

    def __pow__(self, other):
        return GRAElement(pow_f4096(self.value, other.value))

    def __repr__(self):
        return f"GRAElement({repr(self.value)})"

# --- Fibonacci using Float4096 ---

def fibonacci(n):
    """Compute nth Fibonacci number using Binet's formula with high precision"""
    sqrt5 = sqrt(Float4096(5))
    phi = (Float4096(1) + sqrt5) / Float4096(2)
    psi = (Float4096(1) - sqrt5) / Float4096(2)
    # Binet's formula: (phi^n - psi^n)/sqrt(5)
    return ((pow_f4096(phi, n) - pow_f4096(psi, n)) / sqrt5)

# --- Example usage ---

if __name__ == "__main__":
    print("4096-bit precision Float and Fibonacci Demo")

    # Create some Float4096 numbers
    a = Float4096("1.234567890123456789")
    b = Float4096("2.34567890123456789")

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"sqrt(a) = {sqrt(a)}")

    # GRAElement usage
    x = GRAElement("3.141592653589793238462643383279502884")
    y = GRAElement("2.718281828459045235360287471352662497")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x^y = {x ** y}")

    # Fibonacci number with high precision
    n = 100  # example large n
    fib_n = fibonacci(n)
    print(f"Fibonacci({n}) = {fib_n}")
    # To get integer approximation:
    print(f"Integer approximation of Fibonacci({n}): {int(mpmath.nint(fib_n.val))}")

    # ComplexFloat4096 demo
    c1 = ComplexFloat4096(a, b)
    c2 = ComplexFloat4096(b, a)
    print(f"c1 = {c1}")
    print(f"c2 = {c2}")
    print(f"c1 * c2 = {c1 * c2}")
    print(f"|c1| = {c1.abs()}")
