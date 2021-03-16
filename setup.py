from setuptools import setup, find_packages

setup(
    name='ObjectDetection_detectron2',
    version='1.0.0',
    description='Object detection project for cognite',
    author="hasara, asitha, bimsara, shehan",
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/Has970211/Object_Detection",
    #scripts = ['scripts/inference.py'],
    install_requires=['torch', 'torchvision', 'PyYAML', 'Pillow', 'opencv-python'],
    packages=find_packages(),

)
