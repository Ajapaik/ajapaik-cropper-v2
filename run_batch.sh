#!/bin/bash

for ii in img/*; do
    python3 edge_kernel_single.py $ii
done
