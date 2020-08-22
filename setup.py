from setuptools import find_packages, setup

setup(
    name='pyhalcyon',
    version='0.1.0',
    description=(
        'Halcyon: an accurate basecaller exploiting'
        'an encoder-decoder model with monotonic attention'
    ),
    author='Hiroki Konishi',
    url='https://github.com/relastle/halcyon',
    packages=find_packages(),
    scripts=['bin/halcyon'],
    install_requires=[
        line.strip() for line in open('./requirements.txt').readlines()
    ],
    entry_points={
        'console_scripts': [
            'halcyon=halcyon.console:main'
        ]
    }
)
