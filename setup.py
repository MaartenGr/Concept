from setuptools import setup, find_packages

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1"
]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]

base_packages = [
    "numpy>=1.20.0",
    "hdbscan>=0.8.27",
    "umap-learn>=0.5.0",
    "pandas>=1.1.5",
    "scikit-learn>=0.22.2.post1",
    "tqdm>=4.41.1",
    "sentence-transformers==1.2.0",  # Temporarily until the Tokenizer gets updated
    "pillow>=7.1.2",
    "ftfy>=6.0.3"
]

dev_packages = docs_packages + test_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="concept",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.2.0",
    author="Maarten P. Grootendorst",
    author_email="maartengrootendorst@gmail.com",
    description="Topic Model Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="image nlp topic modeling embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
    },
    python_requires='>=3.6',
)
