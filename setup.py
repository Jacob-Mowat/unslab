from setuptools import setup, find_packages


setup(
    name="unslab",
    version="1.7.9",
    packages=find_packages(
        include=["unslab", "unslab.*"],
    ),
)
