from setuptools import setup, find_packages

setup(
    name="think-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "python-dotenv>=1.0.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python SDK for accessing LLMs from various inference providers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/think-sdk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 