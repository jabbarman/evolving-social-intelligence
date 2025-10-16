from setuptools import setup, find_packages

setup(
    name="evolving-social-intelligence",
    version="0.1.0",
    description="An open-ended evolutionary environment where AI agents develop intelligence through social interaction",
    author="Joseph",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "pygame>=2.5.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
