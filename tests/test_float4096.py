# tests/test_float4096.py
import pytest
import numpy as np
import sympy as sp
import math
from numbers import Number
import pytest
import numpy as np
import sympy as sp
from float4096 import (
    Float4096,
    Float4096Array,
    ComplexFloat4096,
    GRAElement,
    GoldenClassField,
    sqrt,
    exp,
    log,
    log10,
    sin,
    cos,
    pi_val,
    linspace,
    logspace,
    mean,
    stddev,
    abs,
    max,
    D,
    D_x,
    F_x,
    invert_D,
    ds_squared,
    g9,
    R9,
    grad9_r_n,
    Gamma_n,
    D_n,
    T,
    Xi_n,
    psi_9,
    E_n,
    edge_weight,
    Coil,
    Spin,
    Splice,
    Reflect,
    Coil_n,
    Spin_n,
    Splice_n,
    Reflect_n,
    recursive_time,
    frequency,
    charge,
    field_yield,
    action,
    energy,
    force,
    voltage,
    labeled_output,
    field_automorphisms,
    field_tension,
    prepare_prime_interpolation,
    native_zeta,
    native_prime_product,
    compute_spline_coefficients,
    native_cubic_spline,
    P_nb,
    solve_n_beta_for_prime,
)

# Constants for testing
EPSILON = Float4096("1e-20")
PRIME_INTERP = prepare_prime_interpolation()
PI = pi_val

@pytest.fixture(autouse=True)
def setup_globals():
    from float4096 import phi, sqrt5, pi_val, k, r
    global phi, sqrt5, pi_val, k, r
    phi = Float4096((1 + sqrt(Float4096(5))) / 2)
    sqrt5 = sqrt(Float4096(5))
    pi_val = Float4096(math.pi)
    k = Float4096(-1)
    r = Float4096(1)

@pytest.fixture
def prime_interp():
    return PRIME_INTERP

def test_cosmo_fit_integration(prime_interp):
    n, beta = Float4096(2), Float4096(0.5)
    val = D(n, beta, prime_interp=prime_interp)
    assert val.is_finite()
    assert float(val) > 0
    n_est, beta_est, scale_est, _, r_est, k_est = invert_D(val, prime_interp=prime_interp)
    assert n_est is not None
    assert abs(float(n_est) - float(n)) < 1e-5

def test_float4096_arithmetic():
    a = Float4096(2.5)
    b = Float4096(1.5)
    
    # Addition
    assert abs((a + b) - Float4096(4.0)) < EPSILON
    # Subtraction
    assert abs((a - b) - Float4096(1.0)) < EPSILON
    # Multiplication (FFT-based)
    assert abs((a * b) - Float4096(3.75)) < EPSILON
    # Division
    assert abs((a / b) - Float4096(2.5 / 1.5)) < EPSILON
    # Exponentiation
    assert abs((a ** Float4096(2)) - Float4096(6.25)) < EPSILON
    # Square root
    assert abs(sqrt(a) - Float4096(math.sqrt(2.5))) < EPSILON
    # Absolute value
    assert abs(Float4096(-2.5)) == Float4096(2.5)
    
    # Edge cases
    assert Float4096(0) + Float4096(0) == Float4096(0)
    with pytest.raises(ValueError, match="Division by zero"):
        a / Float4096(0)
    with pytest.raises(ValueError, match="Division by near-zero value"):
        a / Float4096("1e-50")

def test_float4096_comparison():
    a = Float4096(2.5)
    b = Float4096(1.5)
    c = Float4096(2.5)
    
    assert a == c
    assert a != b
    assert a > b
    assert b < a
    assert a >= c
    assert b <= a

def test_float4096_conversion():
    a = Float4096(2.5)
    assert abs(float(a) - 2.5) < 1e-15
    assert str(a).startswith("Float4096(2.5")
    assert repr(a).startswith("Float4096(digits=")

def test_complexfloat4096_arithmetic():
    z1 = ComplexFloat4096(Float4096(1), Float4096(2))
    z2 = ComplexFloat4096(Float4096(3), Float4096(4))
    
    # Addition
    assert abs((z1 + z2).real - Float4096(4)) < EPSILON
    assert abs((z1 + z2).imag - Float4096(6)) < EPSILON
    # Multiplication
    assert abs((z1 * z2).real - Float4096(-5)) < EPSILON
    assert abs((z1 * z2).imag - Float4096(10)) < EPSILON
    # Division
    result = z1 / z2
    assert abs(result.real - Float4096(11/25)) < EPSILON
    assert abs(result.imag - Float4096(-2/25)) < EPSILON
    # Conjugate
    assert z1.conjugate() == ComplexFloat4096(Float4096(1), Float4096(-2))
    # Absolute value
    assert abs(z1.abs() - Float4096(math.sqrt(5))) < EPSILON
    # Exponentiation
    z_exp = z1.exp()
    assert abs(z_exp.real - Float4096(math.exp(1) * math.cos(2))) < EPSILON
    assert abs(z_exp.imag - Float4096(math.exp(1) * math.sin(2))) < EPSILON
    # Near-zero division
    with pytest.raises(ValueError, match="Division by near-zero complex magnitude"):
        z1 / ComplexFloat4096(Float4096("1e-50"), Float4096("1e-50"))

def test_float4096array_operations():
    arr = Float4096Array([1, 2, 3])
    assert len(arr) == 3
    assert abs(float(arr[0]) - 1.0) < 1e-15
    
    # Arithmetic
    arr2 = arr + Float4096(1)
    assert all(abs(float(arr2[i]) - (i + 2)) < 1e-15 for i in range(3))
    arr3 = arr * Float4096(2)
    assert all(abs(float(arr3[i]) - (i + 1) * 2) < 1e-15 for i in range(3))
    
    # NumPy ufunc compatibility
    np_arr = np.array(arr)
    assert np.allclose(np_arr, [1, 2, 3])
    
    # Mean and stddev
    assert abs(mean(arr) - Float4096(2)) < EPSILON
    assert abs(stddev(arr) - Float4096(np.std([1, 2, 3], ddof=1))) < EPSILON
    # Abs and max
    assert abs(arr).is_finite()
    assert abs(max(arr) - Float4096(3)) < EPSILON

def test_float4096array_edge_cases():
    empty_arr = Float4096Array([])
    with pytest.raises(ZeroDivisionError):
        mean(empty_arr)
    single_arr = Float4096Array([1])
    assert mean(single_arr) == Float4096(1)
    assert stddev(single_arr) == Float4096(0)

def test_special_functions():
    x = Float4096(1.0)
    
    # Exponential
    assert abs(exp(x) - Float4096(math.exp(1))) < EPSILON
    # Logarithm
    assert abs(log(Float4096(2.718281828459045)) - Float4096(1)) < EPSILON
    # Log10
    assert abs(log10(Float4096(10)) - Float4096(1)) < EPSILON
    # Sine and Cosine
    assert abs(sin(Float4096(0)) - Float4096(0)) < EPSILON
    assert abs(cos(Float4096(0)) - Float4096(1)) < EPSILON
    assert abs(sin(PI) - Float4096(0)) < EPSILON
    assert abs(cos(PI) - Float4096(-1)) < EPSILON
    
    # Edge cases
    with pytest.raises(ValueError, match="Log of non-positive number"):
        log(Float4096(0))
    with pytest.raises(ValueError, match="Square root of negative number"):
        sqrt(Float4096(-1))

def test_prime_functions(prime_interp):
    # Prime product
    assert abs(float(native_prime_product(3)) - (2 * 3 * 5)) < 1e-15
    
    # Prime interpolation
    x = Float4096(2.5)
    p = P_nb(x, prime_interp)
    assert abs(p - Float4096(native_cubic_spline(2.5, [math.log(i + 1) / math.log((1 + math.sqrt(5)) / 2) for i in range(1, 5)], [2, 3, 5, 7]))) < EPSILON
    
    # Solve n_beta
    p_target = 5
    n_beta = solve_n_beta_for_prime(p_target, prime_interp)
    assert abs(P_nb(n_beta, prime_interp) - Float4096(p_target)) < EPSILON

def test_cubic_spline():
    x_points = [0.0, 1.0, 2.0, 3.0]
    y_points = [0.0, 1.0, 4.0, 9.0]
    x = 1.5
    result = native_cubic_spline(x, x_points, y_points)
    assert abs(result - 2.25) < 1e-10
    result_edge = native_cubic_spline(4.0, x_points, y_points)
    assert abs(result_edge - 9.0) < 1e-10

def test_gra_element(prime_interp):
    gra = GRAElement(Float4096(2), prime_interp=prime_interp)
    assert gra._value.is_finite()
    assert float(gra) > 0
    
    # Recursive construction
    gra_prev = GRAElement(Float4096(1), prime_interp=prime_interp)
    gra_recursive = GRAElement.from_recursive(Float4096(2), gra_prev, prime_interp=prime_interp)
    assert abs(gra._value - gra_recursive._value) < EPSILON
    
    # GRA operations
    gra2 = GRAElement(Float4096(3), prime_interp=prime_interp)
    gra_sum = gra.gra_add(gra2)
    assert abs(gra_sum - sqrt(gra._value ** 2 + gra2._value ** 2)) < EPSILON
    gra_mult = gra2.gra_multiply(gra)
    assert abs(gra_mult._value - gra2._value) < EPSILON
    
    # Large n
    gra_large = GRAElement(Float4096(1500), prime_interp=prime_interp)
    assert gra_large._value == Float4096(0)

def test_field_computations(prime_interp):
    n = Float4096(2)
    beta = Float4096(0.5)
    val = D(n, beta, prime_interp=prime_interp)
    assert val.is_finite()
    assert float(val) > 0
    
    x = Float4096(2)
    s = sp.Rational(1, 2)
    d_x = D_x(x, s, prime_interp)
    assert isinstance(d_x, ComplexFloat4096)
    
    f_x = F_x(x, s, prime_interp)
    assert abs(f_x.abs() - (d_x.abs() * native_zeta(s).abs())) < EPSILON
    
    # Invert D
    n_est, beta_est, scale_est, err, r_est, k_est = invert_D(val, prime_interp=prime_interp)
    if n_est is not None:
        assert abs(D(n_est, beta_est, r=r_est, k=k_est, scale=scale_est, prime_interp=prime_interp) - val) < Float4096("1e-10") * val

def test_field_automorphisms_and_tension(prime_interp):
    x = Float4096(1)
    s = sp.Rational(1, 2)
    F_val = F_x(x, s, prime_interp)
    autos = field_automorphisms(F_val, x, s, prime_interp)
    assert abs(autos["F_x(s)"] - F_val) < EPSILON
    assert abs(autos["F_x(1-s)"] - Splice(x, s, prime_interp)) < EPSILON
    assert abs(autos["F_-x(s)"] - Reflect(x, s, prime_interp)) < EPSILON
    assert abs(autos["conjugate(F)"] - F_val.conjugate()) < EPSILON
    
    C_val = Float4096(1)
    m_val = Float4096(1)
    tension = field_tension(F_val, C_val, m_val, s)
    assert tension.is_finite()

def test_meta_operators(prime_interp):
    x = Float4096(1)
    s = sp.Rational(1, 2)
    
    # Coil
    coil = Coil(x)
    assert abs(coil.abs() - Float4096(1)) < EPSILON
    
    # Spin, Splice, Reflect
    spin = Spin(x, s, prime_interp)
    splice = Splice(x, s, prime_interp)
    reflect = Reflect(x, s, prime_interp)
    assert abs(spin - F_x(x, s, prime_interp).conjugate()) < EPSILON
    assert abs(splice - F_x(x, 1 - s, prime_interp)) < EPSILON
    assert abs(reflect - F_x(-x, s, prime_interp)) < EPSILON
    
    # Iterative application
    spin_n = Spin_n(x, 2, s, prime_interp)
    assert abs(spin_n - Spin(Spin(x, s, prime_interp).real, s, prime_interp)) < EPSILON

def test_morphing_scale_wrappers():
    n = Float4096(2)
    m_val = Float4096(1)
    
    output = labeled_output(n, m_val)
    assert abs(output["Hz"] - frequency(n)) < EPSILON
    assert abs(output["Time s"] - recursive_time(n)) < EPSILON
    assert abs(output["Charge C"] - charge(n)) < EPSILON
    assert abs(output["Yield Ω"] - field_yield(n, m_val)) < EPSILON
    assert abs(output["Action h"] - action(n, m_val)) < EPSILON
    assert abs(output["Energy E"] - energy(n, m_val)) < EPSILON
    assert abs(output["Force F"] - force(n, m_val)) < EPSILON
    assert abs(output["Voltage V"] - voltage(n, m_val)) < EPSILON

def test_golden_class_field(prime_interp):
    s_list = [sp.Rational(1, 2), sp.Rational(1, 3)]
    x_list = [Float4096(1), Float4096(2)]
    field = GoldenClassField(s_list, x_list, prime_interp)
    
    field_dict = field.as_dict()
    assert len(field_dict) == len(s_list) * len(x_list)
    for name, val in field_dict.items():
        assert isinstance(val, ComplexFloat4096)
    
    # Reciprocity check
    for s in s_list:
        for x in x_list:
            s_conj = 1 - s
            key1 = (str(s), float(x))
            key2 = (str(s_conj), float(x))
            prod = field.field_cache[key1] * field.field_cache[key2]
            assert prod.is_finite()

def test_differential_operators(prime_interp):
    n = Float4096(2)
    
    # ds_squared
    assert ds_squared(n, prime_interp=prime_interp).is_finite()
    
    # g9 and R9
    assert g9(n, prime_interp).is_finite()
    assert R9(n, prime_interp).is_finite()
    
    # grad9_r_n
    assert grad9_r_n(n, prime_interp).is_finite()
    
    # Gamma_n and D_n
    assert Gamma_n(n, prime_interp).is_finite()
    assert D_n(n, prime_interp).is_finite()
    
    # T
    X_n = (n, fib_real(n), P_nb(n, prime_interp), GRAElement(n, prime_interp=prime_interp)._value)
    X_new = T(X_n, prime_interp=prime_interp)
    assert abs(X_new[0] - (n + Float4096(1))) < EPSILON
    
    # Xi_n and psi_9
    r_n, omega_n = Xi_n(n, prime_interp)
    assert r_n.is_finite() and omega_n.is_finite()
    psi = psi_9(n, prime_interp=prime_interp)
    assert abs(psi.abs() - Float4096(1)) < EPSILON
    
    # E_n and edge_weight
    r_n, tau_n = E_n(n, prime_interp=prime_interp)
    assert r_n.is_finite() and tau_n == Float4096(1)
    assert edge_weight(n, prime_interp).is_finite()

def test_linspace_logspace():
    start = Float4096(0)
    stop = Float4096(10)
    num = 5
    ls = linspace(start, stop, num)
    assert len(ls) == num
    assert abs(ls[0] - start) < EPSILON
    assert abs(ls[-1] - stop) < EPSILON
    
    lgs = logspace(Float4096(0), Float4096(1), num)
    assert len(lgs) == num
    assert abs(log10(lgs[0]) - Float4096(0)) < EPSILON
    assert abs(log10(lgs[-1]) - Float4096(1)) < EPSILON

def test_cache_management():
    from float4096 import fib_cache, cache_set
    for i in range(15000):
        cache_set(fib_cache, i, Float4096(i))
    assert len(fib_cache) <= 10000
    assert 14000 in fib_cache
    assert 0 not in fib_cache

def test_performance_fft_multiply(benchmark):
    a = Float4096(123456789.123456789)
    b = Float4096(987654321.987654321)
    result = benchmark.pedantic(lambda: a.fft_multiply(b), rounds=10)
    assert result.is_finite()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
