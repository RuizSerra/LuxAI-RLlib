import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multilux",
    version="0.1",
    author="Jaime Ruiz Serra",
    author_email="jrs271828@gmail.com",
    description="RLlib Multi-Agents wrapper for LuxAI Season 1 (kaggle)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RuizSerra/LuxAI-RLlib",
    packages=setuptools.find_packages(),
    install_requires=[
          'numpy',
          'opencv-python-headless',
          'ray[rllib]'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
