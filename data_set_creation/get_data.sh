#!/bin/bash

curl -O https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
tar -xf review_polarity.tar.gz
python3 flatten_dataset.py
python3 construct_datasets.py
