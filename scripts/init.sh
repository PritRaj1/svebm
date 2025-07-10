#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Creating SV-EBM environment...${NC}"

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  - Anaconda: https://www.anaconda.com/products/distribution"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_TYPE=""
if conda info --base | grep -q "miniconda"; then
    CONDA_TYPE="miniconda"
    echo -e "${GREEN}Detected: Miniconda${NC}"
elif conda info --base | grep -q "anaconda"; then
    CONDA_TYPE="anaconda"
    echo -e "${GREEN}Detected: Anaconda${NC}"
else
    echo -e "${YELLOW}Warning: Could not determine conda type, proceeding anyway${NC}"
fi

if conda env list | grep -q "SV-EBM"; then
    echo -e "${YELLOW}Environment 'SV-EBM' already exists. Do you want to remove it and recreate? (y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${GREEN}Removing existing environment...${NC}"
        conda env remove -n SV-EBM -y
    else
        echo -e "${YELLOW}Using existing environment. Activating...${NC}"
        conda activate SV-EBM
        echo -e "${GREEN}Environment activated successfully!${NC}"
        exit 0
    fi
fi

echo -e "${GREEN}Creating new conda environment 'SV-EBM'...${NC}"
conda create -n SV-EBM python=3.11 -y

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Activating environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate SV-EBM
conda install -c conda-forge tmux -y

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}Installing requirements...${NC}"
cd "$(dirname "$0")/.."
python scripts/requirements.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo -e "${GREEN}To activate the environment, run: conda activate SV-EBM${NC}"
else
    echo -e "${RED}Failed to install requirements${NC}"
    exit 1
fi
