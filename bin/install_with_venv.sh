#!/bin/bash -e

read -p "Want to install virtual env named 'venv' in this project ? (y/n)" answer
if [ "$answer" = "y" ]; then
  echo "Installing virtual env..."
    declare VENV_DIR=$(pwd)/venv
    if ! [ -d "$VENV_DIR" ]; then
        python3 -m venv $VENV_DIR
    fi

    source $VENV_DIR/bin/activate
    echo "Installing requirements..."
    pip install -r requirements-developer.txt
    python3 -m ipykernel install --user --name=venv
    echo "Installing pre-commit..."
    make install_precommit
    echo "Installation complete!";
else
  echo "Installation of virtual env aborted!";
fi
