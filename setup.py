"""Sets up the current directory as a python package for easier imports

This is used in conjunction with "pip install -e ."
"""

from setuptools import setup, find_packages

setup(
    name="cvxpy_examples",
    version="0.0.1",
    install_requires=[
        "numpy<2",
        "matplotlib",
        "scipy",
        "cvxpy",
        "clarabel",
        "ecos",
        "jax",
        "jaxlib",
        "pybullet",
    ],
    extras_require={"dev": ["pylint", "black"]},
    description="CVXPY examples",
    author="Daniel Morton",
    author_email="danielpmorton@gmail.com",
    url="https://github.com/danielpmorton/cvxpy_examples",
    packages=find_packages(exclude=["artifacts"]),
)
