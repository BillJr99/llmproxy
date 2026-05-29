from setuptools import find_packages, setup

setup(
    name="llmproxy",
    version="1.0.0",
    description="OpenAI-compatible multi-provider LLM proxy",
    packages=find_packages(),
    package_data={"llmproxy": ["providers.json"]},
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "flask>=3.0.0",
        "requests>=2.31.0",
        "gunicorn>=21.2.0",
    ],
    entry_points={
        "console_scripts": [
            "llmproxy=llmproxy.__main__:main",
        ],
    },
)
