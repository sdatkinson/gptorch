# . configure.bash

VENV_DIR="venv"
if [ -d ${VENV_DIR} ]; then
    rm -rf ${VENV_DIR}
fi
virtualenv ${VENV_DIR} -p python3

. ${VENV_DIR}/bin/activate

pip install -r requirements.txt
pip install -e .
