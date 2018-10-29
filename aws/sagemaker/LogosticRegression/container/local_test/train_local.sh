#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm -rf test_dir/model/*
rm -rf test_dir/output/*

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
