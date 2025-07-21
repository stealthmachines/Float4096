from setuptools import setup, find_packages

setup(
    name='float4096',
    version='0.9.0',
    packages=find_packages(include=['float4096', 'float4096.*']),
    install_requires=[
        'numpy>=1.21.0',
        'sympy>=1.8',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.60.0',
        'joblib>=1.0.0',
        'pyfftw>=0.13.0',
        'base4096 @ git+https://github.com/ZCHGorg/base4096module.git',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-benchmark>=4.0.0',
        ]
    },
    author='Josef Kulovany',
    author_email='stealthmachines@gmail.com',
    description='High-precision base4096 arithmetic with symbolic recursion, cosmological modeling, FFT, and cubic spline interpolation.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/stealthmachines/Float4096',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.8',
    license='Proprietary',
    include_package_data=True,
    zip_safe=False,
)
