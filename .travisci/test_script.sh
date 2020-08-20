#!/bin/bash
set -e
python --version
python -c "import numpy; print('numpy {}'.format(numpy.__version__))"
python -c "import scipy; print('scipy {}'.format(scipy.__version__))"
python -c "import lfpy_forward_models; print('lfpy_forward_models {}'.format(lfpy_forward_models.__version__))"

python setup.py develop

while true; do
    py.test -v lfpy_forward_models/tests/test_module.py --cov-report term --cov=lfpy_forward_models/tests/
    if [ $? -eq 0 ]
    then
        exit 0
        break
    fi
done
