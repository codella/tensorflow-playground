from setuptools import setup, find_packages

setup(
    name="tensorflow-tutorial",
    version="0.1.0",
    description="Comprehensive TensorFlow capabilities testing project",
    author="TensorFlow Learner",
    author_email="user@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.15.0",
        "tensorflow-hub>=0.15.0",
        "tensorflow-datasets>=4.9.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)