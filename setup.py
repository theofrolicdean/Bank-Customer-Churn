from setuptools import find_packages, setup
from typing import List
from constants import *

def get_requirements(file_path: str) -> List[str]:
      requirements = []
      with open(file_path, "r") as file:
            requirements = file.readlines()
      requirements = [req.strip().replace('\n', '') for req in requirements]

      if HYPEN_DOT in requirements:
            requirements.remove(HYPEN_DOT)
      
      return requirements

setup(name="Bank-Customer-Churn",
      version="0.0.1",
      author="Theo",
      author_email="theofrolic@gmail.com",
      packages=find_packages(),
      install_requires=get_requirements(REQUIREMENTS_PATH))