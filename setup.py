from setuptools import setup, find_packages


setup(
    name="unslab",
    version="1.6.7",
    packages=find_packages(
        include=["unslab", "unslab.*"],
    ),
)
