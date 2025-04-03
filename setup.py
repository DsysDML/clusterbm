from setuptools import setup, find_packages

setup(
    name='clusterbm',
    version='0.1.0',
    author='Lorenzo Rosset, Aurélien Decelle, Beatriz Seoane',
    maintainer='Lorenzo Rosset',
    author_email='rosset.lorenzo@gmail.com',
    description='Unsupervised hierarchical clustering using the learning dynamics of the (Restricted) Boltzmann Machine',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DsysDML/clusterbm',
    packages=find_packages(include=['clusterbm', 'clusterbm.*']),
    include_package_data=True,
    python_requires='>=3.10, <3.13',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'clusterbm=clusterbm.cli:main',
        ],
    },
    install_requires=[
        'adabmDCA>=0.3.0',
        'ete3>=3.1.3',
        'scikit-learn>=1.5.2',
    ],
)