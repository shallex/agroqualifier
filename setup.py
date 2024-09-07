from setuptools import setup, find_packages
from pathlib import Path

THIS_DIR = Path(__file__).parent
REQUIREMENTS_DIR = THIS_DIR / "requirements"

def _load_requirements_file(requirements_file: Path):
    with requirements_file.open("r") as file:
        return [line.lstrip() for line in file.readlines()]

requirements = _load_requirements_file(REQUIREMENTS_DIR / "requirements.txt")

setup(
    name="agroqualifier",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
