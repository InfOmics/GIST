from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='GIST',
    version='0.1.0',
    author='Gospel Ozioma Nnadi',
    description='Leveraging Graph Information for Spatially Informed Patient Data Analysis with GIST',
    author_email ='gospelozioma.nnadi@univr.it',
    license = 'MIT',
    packages =find_packages(), #['GIST', 'GIST.utils'],
    install_requires=requirements,
    python_requires='>=3.10',
    zip_safe = False,
    include_package_data = True,
    url='https://github.com/gospelnnadi/GIST',
)