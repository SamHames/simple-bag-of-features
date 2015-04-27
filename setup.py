from setuptools import setup

setup(
    name='bof',

    packages=['BOF'],

    license='All rights reserved (Currently)',

    description='Bag of features image encoding with hierarchical spherical \
                kmeans for large scale dictionary learning and efficient encoding.',

    author='Sam Hames',
    author_email='sam@hames.id.au',

    install_requires=['scikit-learn',
                      'numpy']
                  )
