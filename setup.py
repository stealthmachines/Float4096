from setuptools import setup, find_packages

setup(
    name='float4096',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'base4096',
    ],
    author='Josef Kulovany',
    author_email='stealthmachines@gmail.com',
    description='High-precision floating-point arithmetic with base4096 encoding and Golden Recursive Algebra',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/float4096',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: ATTACHED',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
