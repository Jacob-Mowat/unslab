from setuptools import setup, find_packages


setup(
    name="unslab",
    version="1.9.10",
    packages=find_packages(
        include=["unslab", "unslab.*"],
    ),
)
