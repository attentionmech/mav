from setuptools import setup

# Read the content from README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openmav",  # Users install with `pip install openmav`
    version="0.0.3",
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
    long_description=long_description,  # Add long description
    long_description_content_type="text/markdown",  # Use "text/x-rst" if using reStructuredText
    url="https://github.com/attentionmech/mav",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)

