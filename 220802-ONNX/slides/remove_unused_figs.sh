#!/bin/bash

for image_file in $(ls figures/)
do
  if grep $image_file *.log --count --silent
  then
    echo "File $image_file is used."
  else
    echo "File $image_file is not used."
    mkdir -p removed_figures/figures
    mv "figures/$image_file" "removed_figures/figures/$image_file"
  fi
done 

for image_file in $(ls logos/)
do
  if grep $image_file *.log --count --silent
  then
    echo "File $image_file is used."
  else
    echo "File $image_file is not used."
    mkdir -p removed_figures/logos
    mv "logos/$image_file" "removed_figures/logos/$image_file"
  fi
done 
