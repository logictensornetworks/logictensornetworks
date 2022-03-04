from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='ltn',
    version='2.0',
    packages=find_packages(include=['ltn']),
    install_requires=[
        'tensorflow',
        'numpy'
    ],
    url='https://github.com/logictensornetworks/logictensornetworks',
    download_url='https://github.com/logictensornetworks/logictensornetworks/archive/refs/tags/v2.0.tar.gz',
    license='MIT',
    author='Samy Badreddine',
    author_email='badreddine.samy@gmail.com',
    description='Logic Tensor Networks',
    keywords=['machine-learning','framework','neural-symbolic-computing','fuzzy-logic','tensorflow'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
    ]
)