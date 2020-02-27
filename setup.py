from setuptools import setup, find_packages


PACKAGENAME = "watson_dist"
VERSION = "0.1"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Duncan Campbell",
    author_email="duncanc@andrew.cmu.edu",
    description="Watson Distribution",
    long_description="An implementation of the Dimroth-Watson distribution in Python",
    install_requires=["numpy","scipy","astropy"],
    packages=find_packages(),
    url="https://github.com/duncandc/watson_dist"
)
