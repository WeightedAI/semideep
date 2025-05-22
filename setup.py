from setuptools import setup, find_packages

setup(
    name="semideep",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.50.0",
    ],
    author="Aydin Abedinia",
    author_email="abedinia.aydin@gmail.com",
    description="Distance-based weighting for semi-supervised deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aydin/semideep",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
