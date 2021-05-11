from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='The aim of this project ct isis to ato increase training data for KNMI's existing fog detection algorithm. BSince foggy images are scarse during day, dusk and nighttime, there e is not enough training data to achieve acceptable model performance during these times of the day. Therefore, we aim to increase the training data by generating artificial fog usingg cycle-consistent generative adversarial networks.',
    author='Thomas ten Heuvel',
    license='MIT',
)
