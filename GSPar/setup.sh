#!/bin/bash

# Defining the library directory
LIB_DIR="./lib"
GSPAR_DIR="$LIB_DIR/gspar"

# Checking if the library directory exists
if [ -d "$GSPAR_DIR" ]; then
    echo "The gspar folder already exists. No need to download it again."
else
    # Creating the lib directory if it doesn't exist
    if [ ! -d "$LIB_DIR" ]; then
        mkdir -p "$LIB_DIR"
    fi

    # Navigating to the lib directory
    cd "$LIB_DIR"

    # Cloning the GSParLib repository
    git clone https://github.com/GMAP/GSParLib.git

    # Renaming the cloned folder to gspar
    mv GSParLib gspar

    # Navigating to the gspar directory
    cd gspar

    # Compiling the library
    make
fi
