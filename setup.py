"""Setup file for churn_detection package."""

from setuptools import setup, find_packages

setup(
    name="churn_detection",
    version="0.1",
    description="Telco Customer Churn Detection with Machine Learning",
    author="abljoel",
    author_email="150628431+abljoel@users.noreply.github.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "kaggle",
        "pyarrow",
        "python-dotenv",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)
