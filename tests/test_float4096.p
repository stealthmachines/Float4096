from float4096 import Float4096, Float4096Array, GRAElement, sqrt, exp, log, linspace, mean, D, invert_D

def test_basic_operations():
    a = Float4096(1.5, mode='numpy')
    b = Float4096(2.5, mode='python')
    assert float(a + b) == 4.0
    assert float(a * b) == 3.75

def test_encoding_roundtrip():
    original = Float4096(3.14159)
    encoded = original.to_base4096()
    decoded = Float4096.from_base4096(encoded)
    assert abs(float(original) - float(decoded)) < 1e-12

def test_array_operations():
    arr = Float4096Array([1.0, 2.0, 3.0])
    assert len(arr) == 3
    assert float(arr[0]) == 1.0
    arr2 = arr + Float4096(1.0)
    assert [float(x) for x in arr2] == [2.0, 3.0, 4.0]
    arr3 = arr * Float4096Array([2.0, 2.0, 2.0])
    assert [float(x) for x in arr3] == [2.0, 4.0, 6.0]

def test_numpy_ufunc():
    arr = Float4096Array([1.0, 4.0, 9.0])
    result = sqrt(arr)
    expected = [float(sqrt(Float4096(x))) for x in [1.0, 4.0, 9.0]]
    assert [float(x) for x in result] == expected

def test_linspace():
    arr = linspace(Float4096(0), Float4096(2), 3)
    assert [float(x) for x in arr] == [0.0, 1.0, 2.0]

def test_mean():
    arr = Float4096Array([1.0, 2.0, 3.0])
    assert float(mean(arr)) == 2.0

def test_gra_element_closed_form():
    r1 = GRAElement(1, Omega=1.0, base=2.0)
    expected = sqrt(Float4096(4) * phi * Float4096(1))
    assert abs(float(r1) - float(expected)) < 1e-10
    r2 = GRAElement(2)
    Fn = fib_real(Float4096(2))
    p2 = Float4096(3)  # Second prime
    expected = sqrt(phi * Float4096(1) * Fn * Float4096(4) * Float4096(2) * p2)
    assert abs(float(r2) - float(expected)) < 1e-10

def test_gra_element_recursive():
    r1 = GRAElement(1)
    r2 = GRAElement.from_recursive(2, prev_r_n_minus_1=r1)
    Fn = fib_real(Float4096(2))
    Fn_minus_1 = fib_real(Float4096(1))
    p2 = Float4096(3)
    expected = r1._value * sqrt(Float4096(2) * p2 * (Fn / Fn_minus_1))
    assert abs(float(r2) - float(expected)) < 1e-10

def test_gra_operations():
    r1 = GRAElement(1)
    r2 = GRAElement(2)
    r3 = r2.gra_multiply(r1)
    assert abs(float(r3) - float(GRAElement(2))) < 1e-10
    r_sum = r1.gra_add(r2)
    expected = sqrt(r1._value ** Float4096(2) + r2._value ** Float4096(2))
    assert abs(float(r_sum) - float(expected)) < 1e-10

def test_D_function():
    val = D(Float4096(2), Float4096(0.5))
    Fn_beta = fib_real(Float4096(2.5))
    p_n = Float4096(PRIMES[int(2.5) % len(PRIMES)])
    expected = sqrt(phi * Fn_beta * exp(Float4096(2.5) * log(Float4096(2))) * p_n * Float4096(1)) * (Float4096(1) ** Float4096(1))
    assert abs(float(val) - float(expected)) < 1e-10

def test_invert_D():
    value = D(Float4096(2), Float4096(0.5))
    n, beta, scale, uncertainty, r, k = invert_D(value)
    assert abs(float(n) - 2.0) < 1e-5
    assert abs(float(beta) - 0.5) < 1e-5
