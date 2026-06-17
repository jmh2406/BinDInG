#!/bin/bash


if [ -z "$1" ]; then
    echo "Please give a project name"
    exit 1
fi

projname=$1

mkdir -p "../projects/$projname/"

cp -r ../projects/Example/* ../projects/$projname/

echo "Created new project directory $projname"
