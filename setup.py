from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

setup(
    name="opt_sugar",
    version="0.0.4.4",
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
    long_description=long_description,
    long_description_content_type='text/x-rst'
)
