from setuptools import find_packages, setup

setup(
    name="mlisplacement",
    version="0.1.0",
    author="",
    description="",  # TODO:
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="",  # TODO:
    license="",  # TODO:
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "transformers",
        "datasets",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
        ],
        "infer_opt": [
            "vllm"
        ],
        "struc_extract": [
            "guidance",
            "accelerate",
            "gpustat",
            "vllm"
        ]
    },
)
