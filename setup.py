import re
from setuptools import setup

setup(
    name="pysgt",
    version="1.0.1",
    description="Stochastic Gradient Trees implementation in Python",
    long_description="Stochastic Gradient Trees by Henry Gouk, Bernhard Pfahringer, and Eibe Frank implementation in Python. Based on the parer's accompanied repository code.",
    url="https://github.com/JoKoum/stochastic-gradient-trees-python",
    author="John Koumentis",
    author_email="jokoum92@gmail.com",
    license="MIT",
    packages=['pysgt','pysgt.utils'],
    install_requires=["numpy>=1.20.2", "scipy>=1.6.2", "pandas>=1.3.3", "scikit-learn>=0.24.2"],
    zip_safe=False
)