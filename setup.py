# -*- coding: utf-8 -*-
from setuptools import setup


setup(
      name='yeaz',
      version=__import__('yeaz').__version__,

      description='YeaZ without a GUI.',
      long_description='YeaZ without a GUI.',
      
      url='https://github.com/prhbrt/yeastcells-detection/',

      author='Herbert Kruitbosch, Yasmin Mzayek',
      author_email='H.T.Kruitbosch@rug.nl, y.mzayek@rug.nl',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
      ],
      keywords='yeast cell detection, microscopy images, tif, tiff, image segmentation, tracking, computer vision',
      
      packages=['yeastcells'],
      install_requires=[
        'h5py==2.9.0',
        'tensorflow>=1.15.2,<2',
        'scikit-image>=0.17.2',
        'scikit-learn>=0.23.2,<0.24', # some threadpoolctl issues at 0.24
        'opencv-python>=4.4.0.46',
        'opencv-contrib-python>=4.4.0.46',
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'Shapely>=1.7.0'
        'tqdm>=4.51.0',
        'pandas>=1.1.4',
        'nd2reader==3.2.1',
        'openpyxl==3.0.1',
        'munkres==1.1.2',
        'imageio>=2.6.1',
        'Pillow>=6.2.1',
      ],
      zip_safe=True,
)
