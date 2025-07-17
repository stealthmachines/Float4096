import numpy as np
import struct
import base4096
from numbers import Number

# Generate prime numbers for use in D and GRAElement
def generate_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]

PRIMES = generate_primes(104729)[:10000]

# Golden ratio as Float4096
phi = float4096((1 + sqrt(float4096(5))) / float4096(2))

# Cache for fib_real
fib_cache = {}

def fib_real(n):
    if n in fib_cache:
        return fib_cache[n]
    if n > 70:
        return float4096(0)
    try:
        term1 = pow(phi, float4096(n)) / sqrt(float4096(5))
        pi_val = pi()
        term2 = pow(inv(phi), float4096(n)) * cos(pi_val * float4096(n))
        result = term1 - term2
        if not result.is_finite():
            return float4096(0)
        fib_cache[n] = result
        return result
    except Exception:
        return float4096(0)

class Float4096:
    def __init__(self, value=0.0, mode='numpy'):
        assert mode in ('numpy', 'python'), "mode must be 'numpy' or 'python'"
        self.mode = mode
        self._value = np.float64(value) if mode == 'numpy' else float(value)
        self._b4096 = base4096.encode(self._pack_bytes())

    def _pack_bytes(self):
        return self._value.tobytes() if self.mode == 'numpy' else struct.pack('<d', self._value)

    def _unpack_bytes(self, b):
        return np.frombuffer(b, dtype=np.float64)[0] if self.mode == 'numpy' else struct.unpack('<d', b)[0]

    def __float__(self):
        return float(self._value)

    def __repr__(self):
        return f"Float4096({float(self)}, mode='{self.mode}')"

    def __str__(self):
        return f"{float(self)} (base4096: {self._b4096})"

    def to_base4096(self):
        return self._b4096

    def to_int(self):
        return int(float(self))

    def is_finite(self):
        return np.isfinite(float(self))

    @classmethod
    def from_base4096(cls, b4096_str, mode='numpy'):
        raw = base4096.decode(b4096_str)
        val = np.frombuffer(raw, dtype=np.float64)[0] if mode == 'numpy' else struct.unpack('<d', raw)[0]
        return cls(val, mode)

    def __add__(self, other):
        if isinstance(other, Float4096Array):
            return other.__add__(self)
        return Float4096(float(self) + float(other), self.mode)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Float4096Array):
            return Float4096Array([self - x for x in other])
        return Float4096(float(self) - float(other), self.mode)

    def __rsub__(self, other):
        return Float4096(float(other) - float(self), self.mode)

    def __mul__(self, other):
        if isinstance(other, Float4096Array):
            return other.__mul__(self)
        return Float4096(float(self) * float(other), self.mode)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Float4096Array):
            return Float4096Array([self / x for x in other])
        return Float4096(float(self) / float(other), self.mode)

    def __rtruediv__(self, other):
        return Float4096(float(other) / float(self), self.mode)

    def __pow__(self, other):
        return Float4096(float(self) ** float(other), self.mode)

    def __rpow__(self, other):
        return Float4096(float(other) ** float(self), self.mode)

    def __eq__(self, other):
        return float(self) == float(other)

    def __lt__(self, other):
        return float(self) < float(other)

    def __le__(self, other):
        return float(self) <= float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __neg__(self):
        return Float4096(-float(self), self.mode)

    def __abs__(self):
        return Float4096(abs(float(self), self.mode)

    def __hash__(self):
        return hash(float(self))

    def __reduce__(self):
        return (Float4096, (float(self), self.mode))

class Float4096Array:
    def __init__(self, values):
        if isinstance(values, (list, tuple, np.ndarray)):
            self._data = [float4096(v, mode='numpy') if not isinstance(v, Float4096) else v for v in values]
        elif isinstance(values, Float4096Array):
            self._data = values._data.copy()
        else:
            raise ValueError("Float4096Array must be initialized with a list, tuple, numpy array, or Float4096Array")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx] if isinstance(idx, (int, slice)) else Float4096Array([self._data[i] for i in idx])

    def __setitem__(self, idx, value):
        if isinstance(idx, (int, slice)):
            self._data[idx] = float4096(value, mode='numpy') if not isinstance(value, Float4096) else value
        else:
            raise ValueError("Indexing must be int or slice")

    def __array__(self, dtype=None):
        return np.array([float(x) for x in self._data], dtype=dtype if dtype else np.float64)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = [np.array(x) if isinstance(x, Float4096Array) else x for x in inputs]
        inputs = [Float4096Array([x]) if isinstance(x, (Float4096, Number)) else x for x in inputs]
        inputs = [np.array([float(v) for v in x._data]) if isinstance(x, Float4096Array) else x for x in inputs]
        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, np.ndarray):
            return Float4096Array([float4096(x, mode='numpy') for x in result])
        return float4096(result, mode='numpy')

    def __array_function__(self, func, types, args, kwargs):
        args = [np.array(x) if isinstance(x, Float4096Array) else x for x in args]
        args = [np.array([float(v) for v in x._data]) if isinstance(x, Float4096Array) else x for x in args]
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return Float4096Array([float4096(x, mode='numpy') for x in result])
        return float4096(result, mode='numpy')

    def __add__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x + other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x + y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x - other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x - y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([other - x for x in self._data])
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x * other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x * y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x / other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x / y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([other / x for x in self._data])
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([x ** other for x in self._data])
        elif isinstance(other, Float4096Array):
            if len(self) != len(other):
                raise ValueError("Array lengths must match")
            return Float4096Array([x ** y for x, y in zip(self._data, other._data)])
        return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (Float4096, Number)):
            return Float4096Array([other ** x for x in self._data])
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Float4096Array):
            return [x == y for x, y in zip(self._data, other._data)]
        return [x == other for x in self._data]

    def __lt__(self, other):
        if isinstance(other, Float4096Array):
            return [x < y for x, y in zip(self._data, other._data)]
        return [x < other for x in self._data]

    def __le__(self, other):
        if isinstance(other, Float4096Array):
            return [x <= y for x, y in zip(self._data, other._data)]
        return [x <= other for x in self._data]

    def __gt__(self, other):
        if isinstance(other, Float4096Array):
            return [x > y for x, y in zip(self._data, other._data)]
        return [x > other for x in self._data]

    def __ge__(self, other):
        if isinstance(other, Float4096Array):
            return [x >= y for x, y in zip(self._data, other._data)]
        return [x >= other for x in self._data]

    def __repr__(self):
        return f"Float4096Array({[float(x) for x in self._data]})"

    def __str__(self):
        return f"Float4096Array({[float(x) for x in self._data]})"

class GRAElement:
    def __init__(self, n, Omega=float4096(1), base=float4096(2)):
        if not isinstance(n, (int, Float4096)) or float(n) < 1:
            raise ValueError("n must be a positive integer or Float4096 >= 1")
        self.n = float4096(n) if isinstance(n, int) else n
        self.Omega = float4096(Omega)
        self.base = float4096(base)
        self._value = self._compute_r_n()

    def _compute_r_n(self):
        """Compute r_n using the closed-form identity: sqrt(φ * Ω * F_n * base^n * Π_{k=1}^{n} p_k)"""
        try:
            n_int = int(float(self.n))
            Fn = fib_real(self.n)
            if Fn == float4096(0):
                return float4096(0)
            product = float4096(1)
            for k in range(1, n_int + 1):
                idx = k % len(PRIMES)
                product *= float4096(PRIMES[idx])
            val = phi * self.Omega * Fn * pow(self.base, self.n) * product
            if not val.is_finite() or val <= float4096(0):
                return float4096(0)
            return sqrt(val)
        except Exception:
            return float4096(0)

    @classmethod
    def from_recursive(cls, n, prev_r_n_minus_1=None, Omega=float4096(1), base=float4096(2)):
        """Compute r_n using the recursive identity: r_n = r_{n-1} * sqrt(2 * p_n * (F_n / F_{n-1}))"""
        n = float4096(n)
        if float(n) < 1:
            raise ValueError("n must be >= 1")
        if float(n) == 1:
            # Base case: r_1 = sqrt(4 * φ * Ω)
            return cls(n, Omega, base)
        if prev_r_n_minus_1 is None:
            prev_r_n_minus_1 = cls(n - float4096(1), Omega, base)
        Fn = fib_real(n)
        Fn_minus_1 = fib_real(n - float4096(1))
        if Fn == float4096(0) or Fn_minus_1 == float4096(0):
            return cls(n, Omega, base)  # Fallback to closed-form if recursive fails
        idx = int(float(n)) % len(PRIMES)
        p_n = float4096(PRIMES[idx])
        factor = sqrt(float4096(2) * p_n * (Fn / Fn_minus_1))
        r_n = prev_r_n_minus_1._value * factor
        result = cls(n, Omega, base)
        result._value = r_n if r_n.is_finite() and r_n > float4096(0) else result._value
        return result

    def gra_multiply(self, other):
        """GRA multiplication: r_n = r_{n-1} ·_G sqrt(2 * p_n * (F_n / F_{n-1}))"""
        if not isinstance(other, GRAElement):
            raise ValueError("GRA multiplication requires a GRAElement")
        if float(self.n) != float(other.n) + 1:
            raise ValueError("GRA multiplication requires n = m + 1")
        return GRAElement.from_recursive(self.n, prev_r_n_minus_1=other, Omega=self.Omega, base=self.base)

    def gra_add(self, other):
        """GRA addition: r_n ⊕_G r_m = sqrt(r_n² + r_m²)"""
        if not isinstance(other, GRAElement):
            raise ValueError("GRA addition requires a GRAElement")
        result = sqrt(self._value ** float4096(2) + other._value ** float4096(2))
        return float4096(result) if result.is_finite() else float4096(0)

    def __float__(self):
        return float(self._value)

    def __repr__(self):
        return f"GRAElement(n={float(self.n)}, value={float(self._value)}, Omega={float(self.Omega)}, base={float(self.base)})"

    def __str__(self):
        return f"GRAElement({float(self._value)})"

def D(n, beta, r=float4096(1), k=float4096(1), Omega=float4096(1), base=float4096(2), scale=float4096(1)):
    try:
        r_n = GRAElement(n + beta, Omega=Omega, base=base)
        val = scale * r_n._value
        if not val.is_finite() or val <= float4096(0):
            return float4096("1e-30")
        return val * (r ** k)
    except Exception:
        return float4096("1e-30")

def invert_D(value, r=float4096(1), k=float4096(1), Omega=float4096(1), base=float4096(2), scale=float4096(1),
             max_n=float4096(5000), steps=1000):
    candidates = []
    try:
        value_abs = abs(value)
        value_safe = value_abs if value_abs > float4096("1e-30") else float4096("1e-30")
        log_val = log10(value_safe)

        start_sf = max(log_val - float4096(3), float4096(-10))
        end_sf = min(log_val + float4096(3), float4096(10))
        scale_factors = pow(float4096(10), linspace(start_sf, end_sf, 10))

        max_n_val = min(5000, max(1000, int(500 * float(abs(log_val)))))
        steps_val = min(2000, max(500, int(200 * float(abs(log_val)))))

        if float(log_val) > 2:
            n_values = logspace(float4096(0), log10(float4096(max_n_val)), steps_val)
        else:
            n_values = linspace(float4096(0), float4096(max_n_val), steps_val)

        r_values = Float4096Array([0.5, 1.0, 2.0])
        k_values = Float4096Array([0.5, 1.0, 2.0])

        for n in n_values:
            for beta in linspace(float4096(0), float4096(1), 5):
                for dynamic_scale in scale_factors:
                    for r_val in r_values:
                        for k_val in k_values:
                            val = D(n, beta, r_val, k_val, Omega, base, scale * dynamic_scale)
                            if val is not None and val.is_finite():
                                diff = abs(val - value_abs)
                                if diff < float4096("1e-10") * value_abs:
                                    candidates.append((diff, n, beta, dynamic_scale, r_val, k_val))
                                    return n, beta, dynamic_scale, val * float4096("0.01"), r_val, k_val
                            val_inv = D(n, beta, r_val, k_val, Omega, base, scale * dynamic_scale)
                            if val_inv is not None and val_inv.is_finite():
                                val_inv = inv(val_inv)
                                diff = abs(val_inv - value_abs)
                                if diff < float4096("1e-10") * value_abs:
                                    candidates.append((diff, n, beta, dynamic_scale, r_val, k_val))
                                    return n, beta, dynamic_scale, val_inv * float4096("0.01"), r_val, k_val

        if not candidates:
            import logging
            logging.error(f"invert_D: No valid candidates for value {value}")
            return None, None, None, None, None, None

        candidates = sorted(candidates, key=lambda x: float(x[0]))[:5]

        valid_vals = []
        for diff, n, beta, s, r, k in candidates:
            val = D(n, beta, r, k, Omega, base, scale * s)
            if val is None or not val.is_finite():
                val = inv(D(n, beta, r, k, Omega, base, scale * s))
            valid_vals.append(val)

        if len(valid_vals) > 1:
            emergent_uncertainty = stddev(valid_vals)
        elif valid_vals:
            emergent_uncertainty = abs(valid_vals[0]) * float4096("0.01")
        else:
            emergent_uncertainty = float4096("1e-10")

        best = candidates[0]
        return best[1], best[2], best[3], emergent_uncertainty, best[4], best[5]

    except Exception as e:
        import logging
        logging.error(f"invert_D failed for value {value}: {e}")
        return None, None, None, None, None, None

def sqrt(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(np.sqrt(float(v)), mode='numpy') for v in x])
    return float4096(np.sqrt(float(x)), mode='numpy')

def exp(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(np.exp(float(v)), mode='numpy') for v in x])
    return float4096(np.exp(float(x)), mode='numpy')

def log(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(np.log(float(v)), mode='numpy') for v in x])
    return float4096(np.log(float(x)), mode='numpy')

def log10(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(np.log10(float(v)), mode='numpy') for v in x])
    return float4096(np.log10(float(x)), mode='numpy')

def cos(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(np.cos(float(v)), mode='numpy') for v in x])
    return float4096(np.cos(float(x)), mode='numpy')

def pi():
    return float4096(np.pi, mode='numpy')

def inv(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(1.0 / float(v), mode='numpy') for v in x])
    return float4096(1.0 / float(x), mode='numpy')

def linspace(start, stop, num):
    start, stop = float4096(start), float4096(stop)
    num = int(num)
    step = (stop - start) / float4096(num - 1)
    return Float4096Array([start + float4096(i) * step for i in range(num)])

def logspace(start, stop, num):
    start, stop = float4096(start), float4096(stop)
    num = int(num)
    return Float4096Array([exp(x) for x in linspace(log(start), log(stop), num)])

def mean(x):
    if isinstance(x, Float4096Array):
        x = x._data
    total = sum(x)
    return total / float4096(len(x))

def stddev(x):
    if isinstance(x, Float4096Array):
        x = x._data
    mu = mean(x)
    n = len(x)
    if n <= 1:
        return float4096(0)
    squared_diff = sum((xi - mu) ** float4096(2) for xi in x)
    return sqrt(squared_diff / float4096(n - 1))

def percentile(x, q):
    if isinstance(x, Float4096Array):
        x = x._data
    q = [float4096(qi) for qi in (q if isinstance(q, (list, tuple)) else [q])]
    x_sorted = sorted(x, key=float)
    n = len(x_sorted)
    result = []
    for qi in q:
        idx = qi * float4096(n - 1) / float4096(100)
        idx_int = int(float(idx))
        frac = idx - float4096(idx_int)
        if idx_int >= n - 1:
            result.append(x_sorted[-1])
        else:
            result.append(x_sorted[idx_int] + frac * (x_sorted[idx_int + 1] - x_sorted[idx_int]))
    return Float4096Array(result) if len(result) > 1 else result[0]

def max(x, y=None):
    if y is None:
        if isinstance(x, Float4096Array):
            return float4096(max(float(v) for v in x._data), mode='numpy')
        return x
    if isinstance(x, Float4096Array) or isinstance(y, Float4096Array):
        x = x._data if isinstance(x, Float4096Array) else [x] * len(y._data)
        y = y._data if isinstance(y, Float4096Array) else [y] * len(x)
        return Float4096Array([float4096(max(float(a), float(b)), mode='numpy') for a, b in zip(x, y)])
    return float4096(max(float(x), float(y)), mode='numpy')

def min(x, y=None):
    if y is None:
        if isinstance(x, Float4096Array):
            return float4096(min(float(v) for v in x._data), mode='numpy')
        return x
    if isinstance(x, Float4096Array) or isinstance(y, Float4096Array):
        x = x._data if isinstance(x, Float4096Array) else [x] * len(y._data)
        y = y._data if isinstance(y, Float4096Array) else [y] * len(x)
        return Float4096Array([float4096(min(float(a), float(b)), mode='numpy') for a, b in zip(x, y)])
    return float4096(min(float(x), float(y)), mode='numpy')

def abs(x):
    if isinstance(x, Float4096Array):
        return Float4096Array([float4096(abs(float(v)), mode='numpy') for v in x])
    return float4096(abs(float(x)), mode='numpy')
