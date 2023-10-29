from setuptools import setup, find_packages

setup(
    name='routing_lib',
    version='0.0.1',
    license='MIT',
    author="Ludovico Lemma",
    author_email='lwdovico@protonmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/lwdovico/routing-lib',
    keywords='Utils',
    install_requires=[
          'math',
          'matplotlib',
          'numpy',
          'Shapely==1.8.5',
          'Fiona',
          'rtree',
          'pyproj',
          'pygeos',
          'scikit-mobility',
          'geopandas',
          'compress-json',
          'tqdm',
          'igraph',
          'eclipse-sumo==1.15.0',
          'sumolib==1.15.0',
          'libsumo==1.15.0',
          'seaborn',
      ],

)