from setuptools import setup

setup(name='gym_pcgrl',
      version='0.4.0',
      install_requires=['gym==0.21.0', 'numpy>=1.17', 'pillow', 'tensorflow==1.15', 'pyglet', 'pyparsing==2.4.7', 'protobuf==3.20.2', 'stable_baselines==2.10.0', 'IPython'],
      py_modules =[],
      author="Ahmed Khalifa",
      author_email="ahmed@akhalifa.com",
      description="A package for \"Procedural Content Generation via Reinforcement Learning\" OpenAI Gym interface.",
      long_description_content_type="text/markdown",
      url="https://github.com/amidos2006/gym-pcgrl",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ]
)