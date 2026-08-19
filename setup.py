import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scratchml",
    version="1.0.0",
    author="Akash Yadav",
    author_email="akashyadav812733@gmail.com",
    description="A production-grade, zero-dependency Machine Learning and Deep Learning library built from scratch using only NumPy and Pandas.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akashyadav75/scratchml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
)
