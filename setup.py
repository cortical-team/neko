from setuptools import setup, find_packages

setup(
    name="neko",
    version="0.3a1",
    packages=find_packages(),
    install_requires=["tensorflow>=2.3", "torch>=1.6", "numpy", "scipy", "tqdm", "requests"],
    python_requires=">=3.8"
)
