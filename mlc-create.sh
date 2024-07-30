#!/bin/bash

# AIME MLC - Machine Learning Container Management 
# 
# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/mlc 
# 
# This software may be used and distributed according to the terms of the MIT LICENSE 


# Run the second script using the forwarded arguments
mlc.sh create $@
































# Check if the correct number of arguments is provided
<< 'COMMENT'
if [ "$#" -ne 0 ] && [ "$#" -ne 3 ]; then
    echo ""
    echo "ERROR:"
    #printf "Correct use: <container_name> <framework_name> <framework_version> \nExample: pt231aime Pytorch 2.3.1-aime"
    echo -e "Correct Usage: <container_name> <framework_name> <framework_version> \nExample: pt231aime Pytorch 2.3.1-aime"
    echo -e ""
    exit 1
fi
COMMENT

# Capture the arguments
#CONTAINER_NAME=$1
#FRAMEWORK_NAME=$2
#FRAMEWORK_VERSION=$3

# Start the second script using the captured arguments
#./mlc.sh "create" "$@" #"$CONTAINER_NAME" "$FRAMEWORK_NAME" "$FRAMEWORK_VERSION"

# Start the Python script using the captured arguments
# Pass the arguments to the Python script
#python3 mlc.py "create" "$CONTAINER_NAME" "$FRAMEWORK_NAME" "$FRAMEWORK_VERSION"

