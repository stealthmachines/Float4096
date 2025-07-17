## Installation

To install the dependencies listed in `requirements.txt` and set up `float4096` and `cosmo_fit.py`, follow these steps. This assumes you have both `float4096` and `cosmo_fit.py` in your project directory, with `float4096` structured as a package (available at [https://github.com/stealthmachines/float4096](https://github.com/stealthmachines/float4096)) and `cosmo_fit.py` as a standalone script.

### Steps

SINGLE COMMAND (assuming local setup):

 ```bash
pip install git+https://github.com/ZCHGorg/base4096.git && pip install -r requirements.txt && cd float4096 && pip install . && cd ..

Manual Installtion:
1. **Set Up a Virtual Environment** (recommended to isolate dependencies):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Prerequisites:
git clone https://github.com/ZCHGorg/base4096.git
cd base4096
pip install .

Clone the repository:
git clone https://github.com/stealthmachines/float4096.git
cd float4096

Install dependencies:
pip install numpy

Install the package:
pip install .

Usage
Basic Arithmetic
from float4096 import Float4096, Float4096Array

# Scalar operations
a = Float4096(1.5)
b = Float4096(2.5)
print(a + b)  # Float4096(4.0, mode='numpy')
print(a * b)  # Float4096(3.75, mode='numpy')

# Array operations
arr = Float4096Array([1.0, 2.0, 3.0])
print(arr * 2)  # Float4096Array([2.0, 4.0, 6.0])

Golden Recursive Algebra (GRA)
from float4096 import GRAElement

# Compute r_n using closed-form
r1 = GRAElement(1)
print(r1)  # GRAElement(~2.297, Omega=1.0, base=2.0)

# Recursive computation
r2 = GRAElement.from_recursive(2, prev_r_n_minus_1=r1)
print(r2)  # GRAElement(~4.104, Omega=1.0, base=2.0)

# GRA operations
r3 = r2.gra_multiply(r1)  # r_n = r_{n-1} ·_G sqrt(2 * p_n * (F_n / F_{n-1}))
print(r3)  # Matches r2
r_sum = r1.gra_add(r2)  # r_n ⊕_G r_m = sqrt(r_n² + r_m²)
print(r_sum)  # ~4.678

D and invert_D Functions
from float4096 import D, invert_D

# Compute D(n, beta)
val = D(Float4096(2), Float4096(0.5))
print(val)  # High-precision emergent constant

# Invert D to find parameters
n, beta, scale, uncertainty, r, k = invert_D(val)
print(n, beta, scale)  # ~2.0, ~0.5, ~1.0

Testing
Run tests to verify functionality:
python -m unittest float4096/tests/test_float4096.py

Dependencies

Python >= 3.6
numpy
base4096 (available at https://github.com/ZCHGorg/base4096)

Project Structure
float4096/
├── float4096/
│   ├── __init__.py
│   ├── float4096.py
│   └── tests/
│       └── test_float4096.py
├── setup.py
└── README.md

Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub.
License
[ZCHG.org](https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440)
