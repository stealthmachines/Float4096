# Float4096

A high-precision floating-point arithmetic library using native base4096 representation, optimized for Golden Recursive Algebra (GRA) and field-theoretic computations in `cosmo_fit.py`.

## Overview

`float4096` provides `Float4096` for scalar arithmetic (~768-bit precision), `ComplexFloat4096` for complex number operations, `Float4096Array` for vectorized operations, `GRAElement` for emergent constants, and `GoldenClassField` for field computations. It features native FFT-based multiplication, full cubic spline interpolation, and complex arithmetic, minimizing `sympy` and `numpy` overhead. Optimized for `cosmo_fit.py`.

## Installation

**EASY BUTTON:**
```bash
cd "$(pwd)" && (command -v python3 >/dev/null 2>&1 && python3 -m venv venv || python -m venv venv) && ( [ -f venv/bin/activate ] && . venv/bin/activate || . venv/Scripts/activate ) && export PYTHONPATH="$PYTHONPATH:$(pwd)" && pip install --upgrade pip && pip install -r requirements.txt && pip install . && pytest tests/test_float4096.py -v && python cosmo_fit/cosmo_fit.py

Breakdown of the Command
cd "$(pwd)": Ensures the command runs from project_root.
(command -v python3 >/dev/null 2>&1 && python3 -m venv venv || python -m venv venv): Checks for python3, falls back to python, ensuring Python 3.8+ is used to create the virtual environment.
([ -f venv/bin/activate ] && . venv/bin/activate || . venv/Scripts/activate): Activates the virtual environment, using venv/bin/activate for Linux/macOS or venv/Scripts/activate for Windows.
export PYTHONPATH="$PYTHONPATH:$(pwd)": Adds project_root to PYTHONPATH for module imports.
pip install --upgrade pip: Upgrades pip for reliable dependency installation.
pip install -r requirements.txt: Installs dependencies from requirements.txt.
pip install .: Installs the float4096 package using setup.py.
pytest tests/test_float4096.py -v: Runs tests with verbose output.
python cosmo_fit/cosmo_fit.py: Runs cosmo_fit.py.

**FOR THOSE WHO PREFER TO DO IT MANUALLY:**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/stealthmachines/Float4096.git
   cd Float4096
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Save the following in `requirements.txt`:
   ```text
   numpy>=1.20.0
   sympy>=1.9
   scipy>=1.7.0
   base4096
   # Note: latest base4096 is not on PyPI, install it for now from source:
   pip install git+https://github.com/ZCHGorg/base4096.git
   ```
   Install dependencies:
   ```bash
   pip install git+https://github.com/ZCHGorg/base4096.git
   pip install -r requirements.txt
   ```

4. **Install `float4096`**:
   ```bash
   pip install .
   ```

5. **Verify Installation**:
   ```bash
   python -m unittest float4096/tests/test_float4096.py
   ```

## Usage

### Basic Arithmetic
```python
from float4096 import Float4096, ComplexFloat4096, Float4096Array

a = Float4096(1.5)
b = Float4096(2.5)
print(a + b)  # Float4096(4.0)
print(a * b)  # Float4096(3.75) using FFT

c = ComplexFloat4096(a, b)
d = ComplexFloat4096(2.0, 1.0)
print(c + d)  # ComplexFloat4096(3.5 + 3.5i)
print(c / d)  # Complex division

arr = Float4096Array([1.0, 2.0, 3.0])
print(arr * 2)  # Float4096Array([2.0, 4.0, 6.0])
```

### Golden Recursive Algebra
```python
from float4096 import GRAElement, prepare_prime_interpolation

prime_interp = prepare_prime_interpolation()
r1 = GRAElement(1, prime_interp=prime_interp)
print(r1)  # GRAElement(~2.297)
r2 = GRAElement.from_recursive(2, prev_r_n_minus_1=r1, prime_interp=prime_interp)
print(r2)  # GRAElement(~4.104)
```

### Field Computations
```python
from float4096 import GoldenClassField

prime_interp = prepare_prime_interpolation()
s_list = ["0.5 + 14.134725*I", "0.5 + 21.022040*I"]
x_list = [5, 10]
GCF = GoldenClassField(s_list, x_list, prime_interp)
GCF.display()
GCF.reciprocity_check()
```

### Meta-Operators
```python
from float4096 import Coil_n, Spin_n, Splice_n, Reflect_n

x = Float4096(5)
s = sp.sympify("0.5 + 14.134725*I")
prime_interp = prepare_prime_interpolation()
print(Coil_n(x, 3))  # ComplexFloat4096(...)
print(Spin_n(x, 3, s, prime_interp))  # ComplexFloat4096(...)
```

### Morphing Scale Wrappers
```python
from float4096 import labeled_output

n = Float4096(5)
m_val = Float4096(1)
output = labeled_output(n, m_val)
for key, val in output.items():
    print(f"{key}: {float(val)}")
```

## Project Structure
```
Float4096/
├── float4096/
│   ├── __init__.py
│   ├── float4096.py
│   └── tests/
│       └── test_float4096.py
├── setup.py
├── requirements.txt
└── README.md
```

## License
zchg.org License(https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440)
