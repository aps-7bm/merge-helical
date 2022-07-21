from setuptools import setup, find_packages
import os

setup(
    name='merge-helical',
    version=open('VERSION').read().strip(),
    #version=__version__,
    author='Alan Kastengren',
    author_email='akastengren@anl.gov',
    url='https://github.com/aps-7bm/merge-helical',
    packages=find_packages(),
    include_package_data = True,
    scripts=['bin/merge-helical'],
    description='Convert helical scan to format usable by tomopy-cli',
    zip_safe=False,
)

