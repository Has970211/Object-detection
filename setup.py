from setuptools import setup, find_packages

setup(
    name='ObjectDetection_detectron2',
    version='1.0.0',
    description='Object detection project for cognite',
    author="hasara, shehan, asitha, bimsara",
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/Has970211/Object_Detection",
    install_requires=['torch', 'torchvision', 'PyYAML'],
    packages=find_packages()
)
