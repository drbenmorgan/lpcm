Description
-----------

lpcm is a python library for fitting multivariate data patterns with local principal curves (lpc); including simple tools for measuring goodness-of-fit.      

Installation
------------

If you'd like to download and install the latest source you'll need git:

    git clone git://github.com/warwick-epp/lpcm.git

lpcm is known to run using Python 2.6.5 and Python 2.7.1+.

The following instructions apply to installation of lpcm on the Warwick CSC machines,
though these steps should provide all dependencies for an otherwise bare python install. 
                          
0) baseDir is the location of the base directory location where all downloaded
python code packages will be installed. 

1) Download lpcm from github, installed using

    cd baseDir/lpcm/src
    rm -r build  (if this already exists)
    python setup.py install --prefix=${PWD}/build

    setenv PYTHONPATH ${PWD}/build/lib/python2.6/site-packages

2) Download setuptools-0.6c11 from http://pypi.python.org/pypi/setuptools

    cd baseDir
    tar zxvf setuptools-0.6c11.tar.gz
    cd setuptools-0.6c11
    mkdir build
    python setup.py install --root=${PWD}/build

    setenv PYTHONPATH ${PYTHONPATH}:${PWD}

3) Download scikit-learn-0.9 from http://scikit-learn.org/stable
    
    cd baseDir
    tar zxvf scikit-learn-0.9.tar.gz
    cd scikit-learn-0.9
    mkdir build
    python setup.py install --root=${PWD}/build

    setenv PYTHONPATH ${PYTHONPATH}:${PWD}/build/lib.linux-x86_64-2.6

4) Download scitools-0.8 from http://code.google.com/p/scitools
    
    cd baseDir
    tar zxvf scitools-0.8.tar.gz
    cd scitools-0.8
    mkdir build
    python setup.py install --prefix=${PWD}/build
    
    setenv PYTHONPATH ${PYTHONPATH}:${PWD}/lib
    

For typical usage of this library look at the documentation in 
... or at the unit tests found in ....

Support
-------

Please feel free to use [issue tracking](https://github.com/epp-warwick/lpcm/issues) on Github to to submit feature requests or bug reports. Please send merge requests on [Github](http://github.com/epp-warwick).

References
----------

We ask that forked projects retain the reference below and propagate this request. Please also use this reference when using the code for academic purposes.

Einbeck, J., Tutz, G., & Evers, L. (2005), Local principal curves, Statistics and Computing 15, 301-313.

An R implementation is available on [CRAN](http://cran.r-project.org/web/packages/LPCM/index.html)

Copyright
---------

Copyright (c) 2011 Daniel Roythorne

License
-------

[BSD](http://www.opensource.org/licenses/BSD-3-Clause)