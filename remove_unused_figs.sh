#!/bin/bash

# Check if the directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Assign the first argument to the directory variable
directory="$1"

# Check if the provided directory exists
if [ ! -d "$directory" ]; then
    echo "Directory $directory does not exist."
    exit 1
fi

# Change directory to the provided directory
cd "$directory" || exit

# Call make command
make

# Create directory to move images
mkdir -p "figures/unused_figs"

# Loop through each image file in the provided directory (excluding figures/unused_figs)
for image_file in "figures/"*
do
    # Exclude figures/unused_figs directory
    if [[ "$image_file" != "figures/unused_figs" ]]; then
        # Extract filename without directory path
        filename=$(basename "$image_file")

        # Check if the file is mentioned more than once in any log file
        if grep -q -c "$filename" ./*.log
        then
            echo "File $filename is in use."
        else
            echo "File $filename is not in use."
            mv "$image_file" "figures/unused_figs/$filename"
        fi
    fi
done

# Clean up after processing the files
make clean

