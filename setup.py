from setuptools import setup

setup(
    name="openmav",  # Users install with `pip install openmav`
    version="0.0.2",
    py_modules=["openmav"],  # Rename your script to openmav.py
    install_requires=[
        "rich",
        "torch",
        "transformers"
    ],
    entry_points={
        "console_scripts": [
            "mav=openmav:main",  # CLI command remains `mav`, runs `main()` from `openmav.py`
        ],
    },
    author="attentionmech",
    author_email="attentionmech@gmail.com",
    description="Model Activation Visualizer",
    url="https://github.com/attentionmech/mav",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)

