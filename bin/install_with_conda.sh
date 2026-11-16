#!/bin/bash -e

read -p "Want to install conda env named 'choice-learn'? (y/n)" answer
if [ "$answer" = "y" ]; then
  echo "Installing conda env..."
  conda create -n choice-learn python=3.10 -y
  source $(conda info --base)/etc/profile.d/conda.sh
  conda activate choice-learn
  echo "Installing requirements..."
  pip install -r requirements-developer.txt
  python3 -m ipykernel install --user --name=choice-learn
  conda install -c conda-forge --name choice-learn notebook -y
  echo "Installing pre-commit..."
  make install_precommit
  echo "Installation complete!";
else
  echo "Installation of conda env aborted!";
fi
