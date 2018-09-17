#!/bin/bash
# Preprocess script for the Messidor-2 data set downloaded from http://latim.univ-brest.fr/messidor2/
# in September 2018

# Assumes that the dataset (IMAGES.part<i>.rar files) resides in ./data/messidor2.
# Assumes annotation file is in ./vendor/messidor.

messidor_dir="./data/messidor2"
vendor_messidor_dir="./vendor/messidor"
messidor_volume1_path="$messidor_dir/IMAGES.part1.rar"
default_output_dir="$messidor_dir/bin2"
# gradability grades is only from messidor-1 dataset
grad_grades_file="messidor_gradability_grades.csv"
dr_grades_file="messidor_data.csv"

print_usage()
{
  echo ""
  echo "Extracting and preprocessing script for Messidor-2 dataset."
  echo ""
  echo "Optional parameters: --only_gradable --large_diameter"
  echo "--only_gradable Skip ungradable images. (default: false)"
  echo "--large_diameter  diameter of fundus to 512 pixels (default: false, 299 pixels)"
  echo "--output_dir 	Path to output directory (default: $default_output_dir)"
  exit 1
}

check_parameters()
{
  if [ "$1" -ge 4 ]; then
    echo "Illegal number of parameters".
    print_usage
  fi
  if [ "$1" -ge 1 ]; then
    for param in $2; do
      if [ $(echo "$3" | grep -c -- "$param") -eq 0 ]; then
        echo "Unknown parameter $param."
        print_usage
      fi
    done
  fi
  return 0
}

if echo "$@" | grep -c -- "-h" >/dev/null; then
  print_usage
fi

strip_params=$(echo "$@" | sed "s/--\([a-z_]\+\)\(=\([^ ]\+\)\)\?/\1/g")
check_parameters "$#" "$strip_params" "output_dir only_gradable large_diameter"

# Get output directory from parameters.
output_dir=$(echo "$@" | sed "s/.*--output_dir=\([^ ]\+\).*/\1/")

# Check if output directory is valid.
if ! [[ "$output_dir" =~ ^[^-]+$ ]]; then
  output_dir=$default_output_dir
fi

if ls "$output_dir" >/dev/null 2>&1; then
  echo "Dataset is already located in $output_dir."
  echo "Specify another output directory with the --output_dir flag."
  exit 1
fi

# Confirm the annotations file is present.
if [ ! -f "$vendor_messidor_dir/$dr_grades_file" ]; then
  echo "$vendor_messidor_dir does not contain $dr_grades_file file with labels!"
  exit 1
fi

count_files=$(ls $messidor_dir/IMAGES | egrep 'png|JPG' 2>/dev/null | wc -l)

if [[ $count_files -ne 1748 ]]; then
  # Check if unrar has been installed.
  dpkg -l | grep unrar 2>&1 1>/dev/null
  if [ $? -gt 0 ]; then
    echo "Please install unrar: apt-get/yum install unrar" >&2
    exit 1
  fi

  if [[ $count_files -ne 1748 ]]; then
    echo "Messidor-2 wasn't unpacked properly before, there are $count_files (expected: 1748)"
  fi

  # Confirm the IMAGES.part<i>.rar files (i=1..4) and annotations .csv files are present.
  rar_count=$(find "$messidor_dir" -maxdepth 1 -iname "IMAGES.part*.rar" | wc -l)

  if [ $rar_count -ne 4 ]; then
    echo "$messidor_dir does not contain all IMAGES.part<i>.rar files! There are $rar_count files (expected: 4)"
    exit 1
  fi

  echo "Unpacking .rar files"
  unrar x -y "$messidor_volume1_path" "$messidor_dir" 1>/dev/null || exit 1
fi

# Copying labels file from vendor to data directory.
cp "$vendor_messidor_dir/$dr_grades_file" "$messidor_dir/$dr_grades_file"

# Preprocess the data set and categorize the images by labels into
#  subdirectories.
# use 512 pixels diameter images ?
if echo "$@" | grep -c -- "--large_diameter" >/dev/null; then
    echo "Diameter of fundus to 512 pixels."
    python preprocess_messidor2.py --data_dir="$messidor_dir" --large_diameter || exit 1
else
    echo "Diameter of fundus to 299 pixels."
    python preprocess_messidor2.py --data_dir="$messidor_dir" || exit 1
fi

## Preprocess the data set and categorize the images by labels into
##  subdirectories.
#python preprocess_messidor2.py --data_dir="$messidor_dir" || exit 1
#
#echo "Preparing data set..."
#mkdir -p "$output_dir/0" "$output_dir/1"
#
#echo "Moving images to new directories..."
#find "$messidor_dir/0" -iname "*.jpg" -exec mv {} "$output_dir/0/." \;
#find "$messidor_dir/1" -iname "*.jpg" -exec mv {} "$output_dir/1/." \;
#
## Convert the data set to tfrecords.
#echo "Converting data set to tfrecords..."
#git submodule update --init
#
#python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir" \
#       --num_shards=2 || \
#    { echo "Submodule not initialized. Run git submodule update --init";
#      exit 1; }
#
#echo "Cleaning up..."
#rm -r "$messidor_path" "$messidor_dir/Messidor-2" "$messidor_dir/labels.csv"
#
#echo "Done!"
#exit
#
## References:
## [1] http://latim.univ-brest.fr/messidor2/
## [2] https://www.kaggle.com/google-brain/messidor2-dr-grades
