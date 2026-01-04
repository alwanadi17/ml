from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    Fungsi ini bakal ngebaca requirements.txt lo
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        # Buat ngehapus "-e ." kalau ada di requirements.txt
        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Alwan Adiuntoro',
    author_email='alwanadiuntoro@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)