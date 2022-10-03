from setuptools import setup, find_packages


setup(
    name="opt_sugar",
    version="0.0.2",
    author="Juan Chacon",
    author_email="juandados@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/juandados/opt-sugar",
    keywords="optimization operations mathematical programming",
    install_requires=[
        "numpy",
        "gurobipy",
        "scikit-learn",
    ],
)
