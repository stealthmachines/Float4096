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
├── tests/
│      └── test_float4096.py
├── cosmo_fit/
│     └── cosmo_fit.py
├── setup.py
├── requirements.txt
└── README.md
```

## Reinstall or Upgrade to Latest
```
 pip install --force-reinstall git+https://github.com/stealthmachines/Float4096.git
```

## Uninstall

1. Single-Line Uninstall Command
This command removes the float4096 package, deactivates and deletes the virtual environment, and optionally removes the cloned repository directory. It assumes the user is in the Float4096 directory after running the Git clone command.

Linux/macOS (Bash/Zsh):
```
cd Float4096 && [ -d venv ] && . venv/bin/activate && pip uninstall -y float4096 && deactivate && cd .. && rm -rf Float4096
```
Windows (Git Bash/WSL):
```
cd Float4096 && [ -d venv ] && . venv/Scripts/activate && pip uninstall -y float4096 && deactivate && cd .. && rm -rf Float4096
```
Windows (PowerShell):
```
cd Float4096; if (Test-Path venv) { . .\venv\Scripts\Activate.ps1; pip uninstall -y float4096; deactivate; cd ..; Remove-Item -Recurse -Force Float4096
```

Breakdown:

cd Float4096: Navigates to the project directory.
[ -d venv ] && . venv/bin/activate (or if (Test-Path venv) { . .\venv\Scripts\Activate.ps1 }): Activates the virtual environment if it exists.
pip uninstall -y float4096: Uninstalls the float4096 package without prompting.
deactivate: Deactivates the virtual environment.
rm -rf venv (or Remove-Item -Recurse -Force venv): Deletes the virtual environment.
cd .. && rm -rf Float4096 (or cd ..; Remove-Item -Recurse -Force Float4096): Moves up one directory and deletes the cloned repository.


## License
zchg.org License(https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440)
