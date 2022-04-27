#!/bin/bash

set -o errexit

if [ ${PWD} = ${BASH_SOURCE[0]} ]; then
    cd ..
fi

: ${PYTHON_EXECUTABLE:=python3}

rm -rf .eggs *.egg-info _skbuild dist
${PYTHON_EXECUTABLE} setup.py sdist -v -v -v
cd dist
sha256sum *
gpg --detach-sign -a *
# twine upload *
