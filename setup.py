import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Gaussian_Decomp",
    author="Hannah Collier",
    description="A package for finding oscillations in timeseries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hannahc243/Gaussian_Decomp.git",
    packages=setuptools.find_packages(),
)