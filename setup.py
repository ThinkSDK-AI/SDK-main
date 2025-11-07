from setuptools import setup, find_packages

setup(
    name="fourier-sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "python-dotenv>=1.0.0",
        "beautifulsoup4>=4.12.0"
    ],
    author="Fourier AI",
    author_email="contact@fourier.ai",
    description="A unified Python SDK for accessing multiple LLM providers with standardized responses, function calling, and internet search capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ThinkSDK-AI/SDK-main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
