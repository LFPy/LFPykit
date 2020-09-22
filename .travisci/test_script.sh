#!/bin/bash
set -e
python --version
python -c "import numpy; print('numpy {}'.format(numpy.__version__))"
python -c "import scipy; print('scipy {}'.format(scipy.__version__))"
python -c "import lfpykit; print('lfpykit {}'.format(lfpykit.__version__))"

python setup.py develop

while true; do
    py.test -v lfpykit/tests/ --cov-report term --cov=lfpykit/tests/
    coverage report --show-missing
    if [ $? -eq 0 ]
    then
        exit 0
        break
    fi
done
