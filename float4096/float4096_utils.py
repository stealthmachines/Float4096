# float4096/float4096_utils.py
import math
import numpy as np
import pickle
import os
import functools
from .float4096_mp import Float4096, sqrt, log, cos, pow_f4096

# Constants
BASE = 4096
DIGITS_PER_NUMBER = 64
PRIMES_FILE = "primes_cache.pkl"

def generate_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]

if os.path.exists(PRIMES_FILE):
    with open(PRIMES_FILE, "rb") as f:
        PRIMES = pickle.load(f)
else:
    PRIMES = generate_primes(104729)[:10000]
    with open(PRIMES_FILE, "wb") as f:
        pickle.dump(PRIMES, f)

def prepare_prime_interpolation(primes_list=PRIMES):
    cache_file = "prime_interp_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            recursive_index_phi = pickle.load(f)
    else:
        indices = [float(i + 1) for i in range(len(primes_list))]
        phi_float = float((1 + math.sqrt(5)) / 2)
        recursive_index_phi = [math.log(i + 1) / math.log(phi_float) for i in indices]
        with open(cache_file, "wb") as f:
            pickle.dump(recursive_index_phi, f)

    @functools.wraps(native_cubic_spline)
    def prime_interp(x):
        return native_cubic_spline(x, recursive_index_phi, primes_list)
    return prime_interp

def fib_real(n: Float4096) -> Float4096:
    """Native base4096 Fibonacci using Binet's formula"""
    n_float = float(n)
    if n_float > 70:
        return Float4096(0)
    from .float4096 import fib_cache, cache_set, phi, sqrt5, pi_val
    if n_float in fib_cache:
        return fib_cache[n_float]
    try:
        term1 = pow_f4096(phi, n) / sqrt5
        term2 = pow_f4096(Float4096(1) / phi, n) * cos(pi_val * n)
        result = term1 - term2
        if not result.isfinite():
            result = Float4096(0)
        cache_set(fib_cache, n_float, result)
        return result
    except Exception:
        return Float4096(0)

def native_prime_product(n: int, prime_interp=None) -> Float4096:
    if n < 0 or n > len(PRIMES):
        raise ValueError(f"n must be between 0 and {len(PRIMES)}")
    from .float4096 import prime_product_cache, cache_set
    if n in prime_product_cache:
        return prime_product_cache[n]
    product = Float4096(1)
    for p in PRIMES[:n]:
        product *= Float4096(p)
    cache_set(prime_product_cache, n, product)
    return product

def P_nb(n_beta: Float4096, prime_interp) -> Float4096:
    n_float = float(n_beta)
    from .float4096 import prime_cache, cache_set
    if n_float in prime_cache:
        return prime_cache[n_float]
    result = Float4096(prime_interp(n_float))
    cache_set(prime_cache, n_float, result)
    return result

def native_cubic_spline(x: float, x_points: list, y_points: list) -> float:
    from .float4096 import compute_spline_coefficients, spline_cache, cache_set
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
