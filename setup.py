from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR / "requirements" / "common.txt") as f:
    common_required = f.read().splitlines()

with open(THISDIR / "requirements" / "dev.txt") as f:
    dev_required = f.read().splitlines()

with open(THISDIR / "README.md", encoding="utf-8") as f:
    long_description = f.read()

main_ns = {}
with open(THISDIR / "boneik" / "__version__.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="boneik-solve-ik",
    author="Christoph Heindl",
    description="Fitting kinematic parameters to best align with set of noisy anchor points in Python.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=main_ns["__version__"],
    packages=find_packages(".", include="boneik*"),
    install_requires=common_required,
    zip_safe=False,
    extras_require={
        "dev": dev_required,
    },
)