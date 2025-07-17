float4096
A high-precision floating-point arithmetic library with support for base4096 encoding and Golden Recursive Algebra (GRA) for symbolic computations.
Overview
float4096 is a Python module designed for high-precision numerical computations, integrating with numpy for array operations and base4096 for encoding. It includes a Float4096 class for scalar operations, a Float4096Array class for NumPy-compatible array operations, and a GRAElement class implementing the Golden Recursive Algebra framework for generating emergent constants based on Fibonacci numbers and prime products.
Installation

Installation Command
To install the dependencies listed in requirements.txt and set up float4096 and cosmo_fit.py, use the following steps and command. This assumes you have both float4096 and cosmo_fit.py in your project directory, with float4096 structured as a package (as per the provided markup for https://github.com/stealthmachines/float4096) and cosmo_fit.py as a standalone script.

Steps
Set Up a Virtual Environment (recommended to isolate dependencies):
bash

Collapse

Wrap

Run

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Save requirements.txt:
Place the requirements.txt content (above) in a file named requirements.txt in your project directory.
Install Dependencies:
If base4096 is available on PyPI, run:
bash

Collapse

Wrap

Run

Copy
pip install -r requirements.txt
If base4096 is not on PyPI, install it from the GitHub repository first, then install the remaining dependencies:
bash

Collapse

Wrap

Run

Copy
pip install git+https://github.com/ZCHGorg/base4096.git
pip install -r requirements.txt
Install float4096 Locally (if not published to PyPI):
Navigate to the float4096 directory (e.g., after cloning https://github.com/stealthmachines/float4096):
bash

Collapse

Wrap

Run

Copy
cd float4096
pip install .
cd ..
Verify Installation:
Test float4096:
bash

Collapse

Wrap

Run

Copy
python -m unittest float4096/tests/test_float4096.py
Run cosmo_fit.py to ensure it works:
bash

Collapse

Wrap

Run

Copy
python cosmo_fit.py
(Ensure categorized_allascii.txt and hlsp_ps1cosmo_panstarrs_gpc1_all_model_v1_lcparam-full.txt are in the directory for cosmo_fit.py.)
Single Command (Assuming Local Setup)
If you’re in the project directory with requirements.txt, float4096/, and cosmo_fit.py, and assuming base4096 needs to be installed from GitHub, use:

bash

Collapse

Wrap

Run

Copy
pip install git+https://github.com/ZCHGorg/base4096.git && pip install -r requirements.txt && cd float4096 && pip install . && cd ..
If float4096 is published to PyPI, include it in requirements.txt (add float4096>=0.1.0) and use:

bash

Collapse

Wrap

Run

Copy
pip install -r requirements.txt
Explanation of Dependencies
For float4096:
numpy>=1.20.0: Required for Float4096Array and numerical operations (e.g., sqrt, exp, log).
base4096: Needed for encoding/decoding Float4096 values. If not on PyPI, it must be installed from https://github.com/ZCHGorg/base4096.git.
For cosmo_fit.py:
pandas>=1.3.0: For handling CODATA and supernova data (parse_categorized_codata, generate_emergent_constants).
scipy>=1.7.0: For optimization (minimize, differential_evolution) and interpolation (interp1d).
matplotlib>=3.4.0: For plotting results (e.g., histograms, scatter plots, supernova fits).
tqdm>=4.60.0: For progress bars in generate_emergent_constants, match_to_codata, etc.
joblib>=1.0.0: For parallel processing in symbolic_fit_all_constants.
base4096: Indirectly required via float4096.
float4096: The core dependency for high-precision arithmetic, GRAElement, D, and invert_D. Installed locally via pip install . or from PyPI if published.
Notes on Deployment

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
MIT License
