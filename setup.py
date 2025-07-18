from setuptools import setup, find_packages

setup(
    name='float4096',
    version='0.7.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'sympy>=1.9',
        'scipy>=1.7.0',
        'base4096',  # Ensure this is pip-installable or bundled
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
    python_requires='>=3.6',
    license='Proprietary',
)
