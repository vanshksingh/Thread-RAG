from setuptools import setup, find_packages

with open("README_CLEAN.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("eval_requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="thread-rag",
    version="1.0.0",
    author="Thread-RAG Contributors",
    description="Sequential context assembly for Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/thread-rag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
)

