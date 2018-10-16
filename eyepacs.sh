#!/bin/bash
# Preprocess script for the EyePACS data set from Kaggle.

# Assumes that the data set resides in ./data/eyepacs.

eyepacs_dir="./data/eyepacs"
vendor_eyepacs_dir="./vendor/eyepacs"
default_pool_dir="$eyepacs_dir/pool"

# default_shuffle_seed=42
default_shuffle_seed=12345
default_partition_seed=12345

default_output_dir="$eyepacs_dir/bin2"
grad_grades="$vendor_eyepacs_dir/eyepacs_gradability_grades.csv"
IMAGE_IDS_FILENAME="image_ids.csv"

# From [1].
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

print_usage()
{
  echo ""
  echo "Extracting and preprocessing script for Kaggle EyePACS."
  echo ""
  echo "Optional parameters: --redistribute, --pool_dir, --seed, --only_gradable --large_diameter"
  echo "--redistribute	Redistribute the data set from pool (default: false)"
  echo "--pool_dir	Path to pool folder (default: $default_pool_dir)"
  echo "--seed		Seed number for shuffling before distributing the data set (default: $default_shuffle_seed)"
  echo "--only_gradable Skip ungradable images. (default: false)"
  echo "--large_diameter  diameter of fundus to 512 pixels (default: false, 299 pixels)"
  echo "--output_dir 	Path to output directory (default: $default_output_dir)"
  exit 1
}

check_parameters()
{
  if [ "$1" -ge 7 ]; then
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

# ‘s/regexp/replacement/flags’
strip_params=$(echo "$@" | sed "s/--\([a-z_]\+\)\(=\([^ ]\+\)\)\?/\1/g")
check_parameters "$#" "$strip_params" "redistribute seed pool_dir only_gradable output_dir large_diameter"

# Get seed from parameters.
shuffle_seed=$(echo "$@" | sed "s/.*--seed=\([0-9]\+\).*/\1/")

# Replace seed with default seed number if no seed number.
if ! [[ "$shuffle_seed" =~ ^-?[0-9]+$ ]]; then
  shuffle_seed=$default_shuffle_seed
fi

# Get pool directory from parameters.
pool_dir=$(echo "$@" | sed "s/.*--pool_dir=\([^ ]\+\).*/\1/")

# Check if output directory is valid.
if ! [[ "$pool_dir" =~ ^[^-]+$ ]]; then
  pool_dir=$default_pool_dir
fi

# Get output directory from parameters.
output_dir=$(echo "$@" | sed "s/.*--output_dir=\([^ ]\+\).*/\1/")

if ! [[ "$output_dir" =~ ^[^-]+$ ]]; then
  output_dir=$default_output_dir
fi

if ls "$pool_dir" >/dev/null 2>&1 && ! echo "$@" | grep -c -- "--redistribute" >/dev/null; then
  echo "Path already exists: $pool_dir."
  echo ""
  echo "If you want to redistribute data sets from the pool, run this "
  echo " with the --redistribute flag."
  echo "If you want to extract and preprocess the images to another pool "
  echo " directory, specify --pool_dir with a non-existing directory."
  exit 1
fi

if ls "$output_dir" >/dev/null 2>&1; then
  echo "Path already exists: $output_dir."
  echo ""
  echo "Specify a non-existing --output_dir if you want to redistribute"
  echo " from the existing pool to another directory, along with "
  echo " the --redistribute flag."
  exit 1
fi

# Skip unpacking if --redistribute parameter is defined.
if ! echo "$@" | grep -c -- "--redistribute" >/dev/null; then
  # Confirm the Basexx .zip files and annotations .xls files are present.
  train_zip_count=$(find "$eyepacs_dir" -maxdepth 1 -iname "train.zip.00*" | wc -l)
  test_zip_count=$(find "$eyepacs_dir" -maxdepth 1 -iname "test.zip.00*" | wc -l)
  train_csv_zip=$(find "$eyepacs_dir" -maxdepth 1 -iname "trainLabels.csv.zip" | wc -l)

  if [ $train_zip_count -ne 5 ]; then
    echo "$eyepacs_dir does not contain all train.zip files!"
    exit 1
  fi

  if [ $test_zip_count -ne 7 ]; then
    echo "$eyepacs_dir does not contain all test.zip files!"
    exit 1
  fi

  if [ $train_csv_zip -ne 1 ]; then
    echo "$eyepacs_dir does not contain trainLabels.csv.zip file!"
    exit 1
  fi

  # Test preprocess script.
  error=$(python preprocess_eyepacs.py -h 2>&1 1>/dev/null)
  if [ $? -ne 0 ]; then
    echo "$error" >&2
    exit 1
  fi

  echo "Unzip the data set (0/2)..."

  # Check if p7zip is installed.
  dpkg -l | grep p7zip-full
  if [ $? -gt 0 ]; then
    echo "Please install p7zip-full: apt-get/yum install p7zip-full" >&2
    exit 1
  fi

  # Unzip training set.
  7z e "$eyepacs_dir/train.zip.001" -o"$pool_dir" || exit 1

  echo "Unzip the data set (1/2)..."

  # Unzip test set.
  7z e "$eyepacs_dir/test.zip.001" -o"$pool_dir" || exit 1

  # Copy test labels from vendor to data set folder.
  cp "$vendor_eyepacs_dir/testLabels.csv.zip" "$eyepacs_dir/."

  # Unzip labels.
  7z e "$eyepacs_dir/trainLabels.csv.zip" -o"$pool_dir" || exit 1
  7z e "$eyepacs_dir/testLabels.csv.zip" -o"$pool_dir" || exit 1

  # use 512 pixels diameter images ?
  if echo "$@" | grep -c -- "--large_diameter" >/dev/null; then
    echo "Diameter of fundus to 512 pixels."
    python preprocess_eyepacs.py --data_dir="$pool_dir" --large_diameter
  else
    echo "Diameter of fundus to 299 pixels."
    python preprocess_eyepacs.py --data_dir="$pool_dir"
  fi

  # Remove images in pool.
  find "$pool_dir" -maxdepth 1 -iname "*.jpeg" -delete

  # Remove ungradable images if needed.
  if echo "$@" | grep -c -- "--only_gradable" >/dev/null; then
    echo "Remove ungradable images."
    cat "$grad_grades" | while read tbl; do
      if [[ "$tbl" =~ ^.*0$ ]]; then
        file=$(echo "$tbl" | sed "s/\(.*\) 0/\1/")
        find "$pool_dir" -iname "$file*" -delete
      fi
    done
  fi
fi

# Distribution numbers for data sets with ungradable images.
if echo "$@" | grep -c -- "--only_gradable" >/dev/null; then
  bin2_0_cnt=39202
  bin2_0_tr_cnt=31106
  bin2_1_tr_cnt=12582
else
  # bin2_0_cnt=48784 # training clase 0 + testing clase 0
  # bin2_0_tr_cnt=40688 # training clase 0
  # bin2_1_tr_cnt=16458 # training clase 1 (total clase 1: 17152)

  # incluyo menos imágenes de clase 0 binaria para balancear el dataset, performance modelo decae ?
  # bin2_0_cnt=38596 # training clase 0 + testing clase 0
  # bin2_0_tr_cnt=30500 # training clase 0

  # 60% clase 0, 40% clase 1
  # bin2_0_cnt=32783 # training clase 0 + testing clase 0
  # bin2_0_tr_cnt=24687 # training clase 0

  # 'incluyo más imágenes clase 0 binaria de kaggle que los noruegos'
  # bin2_0_cnt=58096 # training clase 0 + testing clase 0
  # bin2_0_tr_cnt=50000 # training clase 0

  # cantidades para dataset pequeño en eyepacs_small
  # bin2_0_cnt=160
  # bin2_0_tr_cnt=110
  # bin2_1_tr_cnt=20
  #
  # Nuevo dataset kaggle con proporciones train y test iguales
  #             Total	    clase 0     % 0     clase 1     % 1
  # train	    58473	    45609	    78.0%   12864       22.0%
  # validation	12864	    10291	    80.0%   2573        20.0%
  # test	    8576	    6861	    80.0%   1715        20.0%
  # bin2_0_cnt=62761 # training clase 0 + testing clase 0
  # bin2_0_tr_cnt=55900 # training clase 0 (incluye validacion)
  # bin2_1_tr_cnt=15437 # training clase 1 (total clase 1: 17152)

  #  bin2_0_cnt=42308 # training clase 0 + testing clase 0
  #  bin2_0_tr_cnt=40250 # training clase 0 (incluye validacion)
  #  bin2_1_tr_cnt=16466 # training clase 1 (total clase 1: 17152)

  bin2_0_cnt=42403 # training clase 0 (incluye validation) + testing clase 0
  bin2_0_tr_cnt=35336 # training clase 0 (incluye validation)
  bin2_1_tr_cnt=14293 # training clase 1 (total clase 1: 17152)
fi

echo "Finding images..."
for i in {0..4}; do
  k=$(find "$pool_dir/$i" -iname "*.jpg" | wc -l)
  echo "Found $k images in class $i."
done

# Define distributions for data sets.
bin2_0=$(
find "$pool_dir/"[0-1] -iname "*.jpg" |
shuf --random-source=<(get_seeded_random "$shuffle_seed") |
head -n "$bin2_0_cnt"
)
class_0_cnt=$(echo $bin2_0 | tr " " "\n" | wc -l)
echo "Selected $class_0_cnt binary class 0 images."

bin2_1=$(find "$pool_dir/"[2-4] -iname "*.jpg")
# [pablo] next line is the same as bin2_1_cnt=$(echo $bin2_1 | wc -w)
bin2_1_cnt=$(echo $bin2_1 | tr " " "\n" | wc -l)
echo "Selected $bin2_1_cnt binary class 1 images."

echo "Creating directories for data sets"
mkdir -p "$output_dir/train/0" "$output_dir/train/1"
mkdir -p "$output_dir/test/0" "$output_dir/test/1"
mkdir -p "$output_dir/validation"

distribute_images()
{
  echo "$1" |
  tr " " "\n" |
  $2 -n "$3" |
  xargs -I{} cp "{}" "$4"
}

echo "Gathering $bin2_0_tr_cnt images for train+validation set (0/2)"
distribute_images "$bin2_0" head "$bin2_0_tr_cnt" "$output_dir/train/0/."
# cnt=$(find "$output_dir/train/0/" -name "*.jpg" | wc -l)
# echo "$cnt images copied to $output_dir/train/0/"

echo "Gathering $bin2_1_tr_cnt images for train+validation set (1/2)"
distribute_images "$bin2_1" head "$bin2_1_tr_cnt" "$output_dir/train/1/."
# cnt=$(find "$output_dir/train/1/" -name "*.jpg" | wc -l)
# echo "$cnt images copied to $output_dir/train/1/"

echo "Gathering $(expr $bin2_0_cnt - $bin2_0_tr_cnt) images for test set (0/2)"
distribute_images "$bin2_0" tail "$(expr $bin2_0_cnt - $bin2_0_tr_cnt)" "$output_dir/test/0/."
# cnt=$(find "$output_dir/test/0/" -name "*.jpg" | wc -l)
# echo "$cnt images copied to $output_dir/test/0/"

echo "Gathering $(expr $bin2_1_cnt - $bin2_1_tr_cnt) images for test set (1/2)"
distribute_images "$bin2_1" tail "$(expr $bin2_1_cnt - $bin2_1_tr_cnt)" "$output_dir/test/1/."
# cnt=$(find "$output_dir/test/1/" -name "*.jpg" | wc -l)
# echo "$cnt images copied to $output_dir/test/1/"

echo "Converting data set to tfrecords..."
git submodule update --init

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir/train" \
       --num_shards=8 --validation_size=0.2 --random_seed="$default_partition_seed"|| \
    { echo "Submodule not initialized. Run git submodule update --init";
      exit 1; }

echo "Moving validation tfrecords to separate folder."
find "$output_dir/train" -name "validation*.tfrecord" -exec mv {} "$output_dir/validation/." \;
find "$output_dir/train" -maxdepth 1 -iname "*.txt" -exec cp {} "$output_dir/validation/." \;
# image_ids_validation.csv
find "$output_dir/train" -maxdepth 1 -iname "*validation.csv" -exec mv {} "$output_dir/validation/$IMAGE_IDS_FILENAME" \;

python ./create_tfrecords/create_tfrecord.py --dataset_dir="$output_dir/test" \
       --num_shards=4 || exit 1

echo "Done!"
exit

# References:
# [1] https://stackoverflow.com/questions/41962359/shuffling-numbers-in-bash-using-seed
