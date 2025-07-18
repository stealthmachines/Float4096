# project_root/setup.py
from setuptools import setup

setup(
    name='float4096',
    version='0.7.0',
    packages=['float4096'],  # Only include float4096 package
    install_requires=[
        'numpy>=1.21.0',
        'sympy>=1.8',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.60.0',
        'joblib>=1.0.0',
        'base4096 @ git+https://github.com/ZCHGorg/base4096.git',
        'pyfftw>=0.13.0',  # Optional, but include for consistency
    ],
    author='Josef Kulovany',
    author_email='stealthmachines@gmail.com',
    description='High-precision base4096 arithmetic with FFT multiplication, cubic spline, and complex support',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/stealthmachines/Float4096',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    license='Proprietary',
)
