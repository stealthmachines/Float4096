# float4096_mp.py
import mpmath

# Set precision to ~4096 bits (~1234 decimal digits)
mpmath.mp.dps = 1234  # decimal places, roughly 4096 bits

class Float4096:
    def __init__(self, value):
        # Accept int, float, str, or another Float4096
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

    # Comparison operators for completeness
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


class ComplexFloat4096:
    def __init__(self, real, imag=0):
        # Accept Float4096 or numeric for real and imag parts
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
        # sqrt(real^2 + imag^2)
        return Float4096(mpmath.sqrt(self.real.val**2 + self.imag.val**2))

    def __repr__(self):
        return f"ComplexFloat4096({repr(self.real)}, {repr(self.imag)})"


class Float4096Array:
    def __init__(self, values):
        # values: list or iterable of Float4096 or numeric
        self.values = [v if isinstance(v, Float4096) else Float4096(v) for v in values]

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    # Example elementwise addition
    def __add__(self, other):
        if isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Length mismatch in Float4096Array addition")
            return Float4096Array([a + b for a, b in zip(self.values, other.values)])
        else:
            return Float4096Array([a + other for a in self.values])

    def __repr__(self):
        return f"Float4096Array([{', '.join(str(v) for v in self.values)}])"


# Math functions using mpmath, accepting Float4096 or Float4096Array

def sqrt(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.sqrt(x.val))
    elif isinstance(x, Float4096Array):
        return Float4096Array([sqrt(v) for v in x])
    else:
        return Float4096(mpmath.sqrt(mpmath.mpf(x)))

def exp(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.exp(x.val))
    elif isinstance(x, Float4096Array):
        return Float4096Array([exp(v) for v in x])
    else:
        return Float4096(mpmath.exp(mpmath.mpf(x)))

def log(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.log(x.val))
    elif isinstance(x, Float4096Array):
        return Float4096Array([log(v) for v in x])
    else:
        return Float4096(mpmath.log(mpmath.mpf(x)))

def sin(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.sin(x.val))
    elif isinstance(x, Float4096Array):
        return Float4096Array([sin(v) for v in x])
    else:
        return Float4096(mpmath.sin(mpmath.mpf(x)))

def cos(x):
    if isinstance(x, Float4096):
        return Float4096(mpmath.cos(x.val))
    elif isinstance(x, Float4096Array):
        return Float4096Array([cos(v) for v in x])
    else:
        return Float4096(mpmath.cos(mpmath.mpf(x)))

def pow_f4096(base, exponent):
    # base and exponent are Float4096 or numeric
    base_val = base.val if isinstance(base, Float4096) else mpmath.mpf(base)
    exp_val = exponent.val if isinstance(exponent, Float4096) else mpmath.mpf(exponent)
    return Float4096(mpmath.power(base_val, exp_val))
