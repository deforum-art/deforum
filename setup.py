from setuptools import setup, find_packages

setup(
    name='deforum',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='An example Python package',
    long_description=open('README.md').read(),
    install_requires=[],
    url='https://github.com/deforum-art/deforum',
    author='deforum',
    author_email='deforum.art@gmail.com'
)