from setuptools import find_packages, setup

setup(
    name='WaveTorch',
    version='0.1.0',
    author='ZeyuanDong',
    author_email='dongzeyuan@mail.ioa.ac.cn',
    description='A PyTorch-based library for efficient frequency-domain Helmholtz equation solving using Modified Born Series (MBS).',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['torch>=2.0', 'matplotlib'],
    python_requires='>=3.8',
    license='LGPL-3.0-or-later',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Imaging',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
