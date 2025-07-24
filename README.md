# The full of this project is in ALPHA. Use at your own risk!!

# Float4096

A high-precision floating-point arithmetic library using native base4096 representation, optimized for Golden Recursive Algebra (GRA) and field-theoretic computations in `cosmo_fit.py`.

## Overview

`float4096` provides `Float4096` for scalar arithmetic (~768-bit precision), `ComplexFloat4096` for complex number operations, `Float4096Array` for vectorized operations, `GRAElement` for emergent constants, and `GoldenClassField` for field computations. It features native FFT-based multiplication, full cubic spline interpolation, and complex arithmetic, minimizing `sympy` and `numpy` overhead. Optimized for `cosmo_fit.py`.

## Installation
```bash
git clone https://github.com/stealthmachines/Float4096.git
cd Float4096
chmod +x setup_project.sh
./setup_project.sh both
```

## Or manually, if you prefer -

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/stealthmachines/Float4096.git
   cd Float4096
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows (Git Bash/WSL): source venv/Scripts/activate
   ```

3. **Install Dependencies (failing 'pip install -e .', below:)**:
   Save the following in `requirements.txt`:
   ```text
   numpy>=1.21.0
   sympy>=1.8
   scipy>=1.7.0
   pandas>=1.3.0
   matplotlib>=3.4.0
   tqdm>=4.60.0
   joblib>=1.0.0
   pyfftw>=0.13.0
   base4096 @ git+https://github.com/ZCHGorg/base4096module.git
   pytest>=7.0.0
   pytest-benchmark>=4.0.0
   ```
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install `float4096`**:
   ```bash
   pip install -e .
   ```

5. **Verify Installation**:
   ```bash
   python -m unittest discover -s tests -v
   ```
   or
   ```bash
   pytest -v > pytest_output.txt
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
print(Spin_n(x, prime_interp=prime_interp))  # ComplexFloat4096(...)
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
│   ├── float4096_mp.py
│   ├── float4096_utils.py
│   └── float4096.py
├── tests/
│   └── test_float4096.py
├── cosmo_fit/
│   └── cosmo_fit.py
├── setup.py
├── requirements.txt
├── setup_project.sh
└── README.md
```

## Reinstall or Upgrade to Latest (Warning: Nuclear)
```
pip install --force-reinstall git+https://github.com/stealthmachines/Float4096.git
```

## Uninstall

### Single-Line Uninstall Command
This command removes the float4096 package and optionally removes the cloned repository directory. It assumes the user is in the Float4096 directory after running the Git clone command. Consult 'breakdown' for environment removal and other options.

**Linux/macOS (Bash/Zsh)**:
```
cd Float4096 && [ -d venv ] && . venv/bin/activate && pip uninstall -y float4096 && deactivate && cd .. && rm -rf Float4096
```

**Windows (Git Bash/WSL)**:
```
cd Float4096 && [ -d venv ] && . venv/Scripts/activate && pip uninstall -y float4096 && deactivate && cd .. && rm -rf Float4096
```

**Windows (PowerShell)**:
```
cd Float4096; if (Test-Path venv) { . .\venv\Scripts\Activate.ps1; pip uninstall -y float4096; deactivate; cd ..; Remove-Item -Recurse -Force Float4096 }
```

### Breakdown
- `cd Float4096`: Navigates to the project directory.
- `[ -d venv ] && . venv/bin/activate` (or `if (Test-Path venv) { . .\venv\Scripts\Activate.ps1 }`): Activates the virtual environment if it exists.
- `pip uninstall -y float4096`: Uninstalls the float4096 package without prompting.
- `deactivate`: Deactivates the virtual environment.
- `cd .. && rm -rf Float4096` (or `cd ..; Remove-Item -Recurse -Force Float4096`): Moves up one directory and deletes the cloned repository.

## Speaking of Environments (for those who messed up their environments, salvation..)

(Replace Owner with your username)

### Steps to Diagnose Environment Crash
If you disable environment creation due to crashes, try recreating my-env manually:

**Delete Existing my-env**:
```
rm -rf /home/Owner/Float4096/my-env  # Linux/macOS
```
**Windows (PowerShell)**:
```
Remove-Item -Recurse -Force C:\Users\Owner\Documents\Float4096\my-env
```

**Create New Virtual Environment**:
```
python3 -m venv /home/Owner/Float4096/my-env  # Linux/macOS
# or
python -m venv C:\Users\Owner\Documents\Float4096\my-env  # Windows
```

**Activate and Upgrade pip**:
**Linux/macOS**:
```
source /home/Owner/Float4096/my-env/bin/activate
pip install --upgrade pip
```
**Windows (PowerShell)**:
```
C:\Users\Owner\Documents\Float4096\my-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Install Dependencies**:
```
pip install numpy>=1.21.0 sympy>=1.8 scipy>=1.7.0 pandas>=1.3.0 matplotlib>=3.4.0 tqdm>=4.60.0 joblib>=1.0.0 pyfftw>=0.13.0 pytest>=7.0.0 pytest-benchmark>=4.0.0
pip install git+https://github.com/ZCHGorg/base4096module.git
```

**More**:
- `rm -rf venv` (or `Remove-Item -Recurse -Force venv`): Deletes the virtual environment.
- `cd .. && rm -rf Float4096` (or `cd ..; Remove-Item -Recurse -Force Float4096`): Moves up one directory and deletes the cloned repository.

## 🧮 Float4096 v9 – Feature Overview

All major capabilities of the `Float4096` framework categorized by function. Fully symbolic, precision-native, and physics-aware.

---

# Click the below arrows to expand -

[details="📦 Core Types & Math"]
| Feature                       | Description                                      | Status |
|------------------------------|--------------------------------------------------|--------|
| `Float4096`                  | Arbitrary-precision real (via `mpmath`)         | ✅     |
| `ComplexFloat4096`           | Complex number support with base4096 precision  | ✅     |
| `Float4096Array`             | NumPy-like container with ufunc overrides       | ✅     |
| High-precision math ops      | `sqrt`, `exp`, `log`, `sin`, `cos`, `pow`, etc. | ✅     |
| `log10`, `abs`, `max`        | Custom overloads for precision types            | ✅     |
[/details]

---

[details="🔁 Recursive Structures"]
| Feature                      | Description                                          | Status |
|-----------------------------|------------------------------------------------------|--------|
| `GRAElement`                | Golden Recursive Algebra number                     | ✅     |
| `from_recursive()`         | Builds `rₙ` from `rₙ₋₁` using φ, primes, and Fₙ     | ✅     |
| `gra_add`, `gra_multiply`  | Symbolic recursive composition operations            | ✅     |
| `GoldenClassField`         | Recursive physical field mapped over `GRAElement`s   | ✅     |
[/details]

---

[details="🔢 Sequences & Interpolation"]
| Feature                    | Description                                            | Status |
|---------------------------|--------------------------------------------------------|--------|
| Fibonacci (Binet’s φ form) | Precision Fibonacci with π-correction                 | ✅     |
| Prime cache                | Cached list of 10,000+ primes                         | ✅     |
| `P_nb(n)`                 | Interpolated prime value at fractional index           | ✅     |
| Prime spline interpolation | Smooth `logφ`-based cubic spline estimator            | ✅     |
[/details]

---

[details="ζ Zeta & Series"]
| Feature           | Description                                  | Status |
|------------------|----------------------------------------------|--------|
| `native_zeta(s)` | Complex ζ(s) series approximation (Euler sum) | ✅     |
| Zeta caching      | LRU-style memoization for reuse              | ✅     |
[/details]

---

[details="⚛ Symbolic Physics Dimensions"]
| Feature                | Description                                     | Status |
|------------------------|-------------------------------------------------|--------|
| `D(n, β, ...)`         | Recursive dimension operator                    | ✅     |
| `invert_D(...)`        | Root-solve inverse of dimension operator        | ✅     |
| Physical units         | `action`, `energy`, `charge`, `force`, etc.    | ✅     |
| Consistency checks     | `ds_squared`, `Gammaₙ`, `grad9_rₙ`, etc.       | ✅     |
[/details]

---

[details="🌀 Field Automorphisms"]
| Feature        | Description                                | Status |
|----------------|--------------------------------------------|--------|
| `Spin`, `Coil` | Symbolic morphisms over recursive fields   | ✅     |
| `Splice`       | Field blend operator                       | ✅     |
| `Reflect`      | Dual-state symmetry operator               | ✅     |
| `field_automorphisms()` | Returns standard field operations | ✅     |
[/details]

---

[details="⏱ Recursive Time & Frequency"]
| Feature             | Description                           | Status |
|--------------------|---------------------------------------|--------|
| `T(n)`, `Xiₙ`       | Recursive time/frequency generators   | ✅     |
| `recursive_time()` | Converts `n` to harmonic time period  | ✅     |
| `field_yield`, `field_tension` | Derivative-like behavior maps | ✅     |
[/details]

---

[details="📈 Interpolation & Spline"]
| Feature                 | Description                               | Status |
|-------------------------|-------------------------------------------|--------|
| `native_cubic_spline()` | φ-scaled spline interpolation             | ✅     |
| `compute_spline_coefficients()` | Calculates cubic coefficients    | ✅     |
| Spline cache            | LRU-memoized spline evaluations           | ✅     |
[/details]

---

[details="⚙️ Utilities & Arrays"]
| Feature         | Description                        | Status |
|----------------|------------------------------------|--------|
| `linspace`, `logspace` | Range generation for Float4096 | ✅     |
| `mean`, `stddev`       | Array-based statistical ops     | ✅     |
| FFT support     | Uses `pyfftw` if available, else NumPy | ✅     |
[/details]

---

[details="💾 Caching & Files"]
| Feature         | Description                                 | Status |
|----------------|---------------------------------------------|--------|
| Prime cache     | Disk-cached with fallback generation        | ✅     |
| Zeta cache      | OrderedDict-based LRU                       | ✅     |
| Spline & fib cache | Intermediate result memoization          | ✅     |
| Auto-load/save  | `.pkl` used for `primes_cache`, spline, etc.| ✅     |
[/details]

---

[details="🌌 Cosmology & Physics Integration"]
| Feature              | Description                                  | Status |
|----------------------|----------------------------------------------|--------|
| `cosmo_fit.py`       | Cosmological dataset fitting using Float4096 | ✅     |
| `labeled_output()`   | Dimension-keyed result bundling              | ✅     |
| Physical constants   | All symbolic units expressed recursively     | ✅     |
[/details]

---

[details="🧪 Dev Readiness"]
| Feature         | Description                        | Status |
|----------------|------------------------------------|--------|
| Modular layout  | `float4096/` with clear `__init__` | ✅     |
| Precision split | `float4096_mp.py` for math core    | ✅     |
[/details]

# TO DO:
The Base4096 module has fallen or devolved out of our suite, but we will be bringing it back in. This 4096 character alphabet unlocks loads of fun, when I finally get it to play nice with our suite. It was pulled out due to padding problems which I was too tired at the time to fix, as I am now. And so, I will do it later on. God bless you.

https://github.com/ZCHGorg/base4096module

## License
zchg.org License (https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440)
