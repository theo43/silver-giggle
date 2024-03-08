import setuptools
import version

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name='translation',
    version=version.VERSION,
    packages=setuptools.find_packages(exclude=['tests', 'tests.*'], where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    author='silver-gigglers',
    include_package_data=True,
    python_requires='>=3.10',
)
