"""Setup configuration for HNSW-CS-328."""
from setuptools import setup, find_packages

setup(
    name="hnsw-cs328",
    version="2.0.0",
    description="Data-driven parameter optimization for HNSW approximate nearest neighbor search",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="CS-328 Team",
    author_email="team@hnsw-cs328.local",
    url="https://github.com/yourusername/hnsw-cs328",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "hnswlib>=0.8.0",
        "optuna>=3.6",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "PyYAML>=6.0",
        "h5py",
        "typer[all]>=0.9.0",
        "mlflow>=2.10.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
            "flake8>=6.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hnsw-optimize=src.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
