from setuptools import setup

setup(
    name='TUDASummerSchool25Code',
    version='0.1.3',    
    description='Support Code for TUDa FL Lab',
    url='https://github.com/TRUST-TUDa/TUDASummerSchool25Code',
    author='TUDa',
    license='BSD 2-clause',
    packages=['TUDASummerSchool25Code'],
    install_requires=['numpy',                     
                      'torch',
                      'datetime', 'Pillow', 'pytorch-lightning'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
