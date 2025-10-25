from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="helicopter-control-tuner",
    version="1.0.0",
    author="Mahdi Sarfarazi",
    author_email="Mahdi_sarfarazi@outlook.com",
    description="Multi-loop helicopter control system tuner replicating MATLAB systune",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahdisf/classic-multiloop-controller",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Control Systems",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "control>=0.9.0",
        "matplotlib>=3.5.0",
        "slycot>=1.9.0",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
    },
)
