from setuptools import setup, find_packages

setup(
    name="well-test-interpretation",
    version="1.0.0",
    description="ML-based well test interpretation and pressure transient analysis",
    author="Ing. Kelvin Cabrera",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "joblib>=1.3",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "well-test-app=app:main",
        ],
    },
)
