from setuptools import setup, find_packages

# Read the content from README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openmav",
    version="0.0.8",
    packages=find_packages(),  # Automatically find packages inside the repo
    install_requires=[
        "rich",
        "torch",
        "transformers"
    ],
    entry_points={
        "console_scripts": [
            "mav=openmav.mav:main",  # Update path since mav.py is inside openmav/
        ],
    },
    author="attentionmech",
    author_email="attentionmech@gmail.com",
    description="MAV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/attentionmech/mav",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
