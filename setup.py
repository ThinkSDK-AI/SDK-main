from setuptools import setup, find_packages

# Read base requirements
with open("requirements-base.txt") as f:
    base_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Optional provider-specific dependencies
extras_require = {
    # AWS Bedrock provider
    "bedrock": [
        "boto3>=1.28.0",
    ],

    # Web search feature (used by all providers)
    "search": [
        "beautifulsoup4>=4.12.0",
    ],

    # All providers combined
    "all": [
        "boto3>=1.28.0",
        "beautifulsoup4>=4.12.0",
    ],

    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
}

setup(
    name="fourier-sdk",
    version="0.2.0",
    packages=find_packages(),
    install_requires=base_requires,
    extras_require=extras_require,
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
    entry_points={
        "console_scripts": [
            "fourier=cli:main",
        ],
    },
)
