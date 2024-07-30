#!/bin/bash

# AIME MLC - Machine Learning Container Management 
# 
# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/mlc 
# 
# This software may be used and distributed according to the terms of the MIT LICENSE 

# Save all arguments in a variable
all_args=$@

# Pass the arguments to the Python script
python3 mlc.py $all_args


