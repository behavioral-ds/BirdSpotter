import setuptools 
from distutils.core import setup
setup(
        name = 'birdspotter',
        packages = ['birdspotter'],
        version = '0.1.6b1',
        license = 'MIT',
        description = 'A package to measure the influence and botness of twitter users, from twitter dumps',
        author = 'Rohit Ram',
        author_email = 'rohitram96@gmail.com',
        url = 'https://github.com/behavioral-ds/BirdSpotter',
        download_url = 'https://github.com/behavioral-ds/BirdSpotter/archive/0.1.6b1.tar.gz',
        keywords = ['twitter', 'influence', 'botness', 'birdspotter', 'dumps'],
        include_package_data=True,
        scripts=['bin/birdspotter'],
        python_requires='>=3',
        setup_requires=['wheel'],
        install_requires=[
            'tqdm',
            'wget',
            'simplejson',
            'numpy',
            'pandas',
            'python-dateutil',
            'pytz',
            'scikit-learn',
            'scipy',
            'six',
            'sklearn',
            'ijson',
            'xgboost==0.81'
        ],
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Sociology',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
    )
