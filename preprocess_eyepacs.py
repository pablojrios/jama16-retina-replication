import argparse
import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import scale_normalize

parser = argparse.ArgumentParser(description='Preprocess EyePACS data set.')
parser.add_argument("--data_dir", help="Directory where EyePACS resides.",
                    default="data/eyepacs")
parser.add_argument("--large_diameter", action="store_true",
                    help="diameter of fundus to 512 pixels.")

args = parser.parse_args()
data_dir = str(args.data_dir)
large_diameter = bool(args.large_diameter)

train_labels = join(data_dir, 'trainLabels.csv')
test_labels = join(data_dir, 'testLabels.csv')

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1, 2, 3, 4]
        if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []

diameter = 512 if large_diameter else 299
print("Large fundus diameter={}".format(large_diameter))

n = 0
for labels in [train_labels, test_labels]:
    with open(labels, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        for i, row in enumerate(reader):
            basename, grade = row[:2]

            im_path = glob(join(data_dir, "{}*".format(basename)))[0]

            # Find contour of eye fundus in image, and scale
            #  diameter of fundus to 299 pixels and crop the edges.
            res = scale_normalize(save_path=tmp_path, 
                                  image_path=im_path,
                                  diameter=diameter, verbosity=0)

            n += 1
            # Status message.
            msg = "\r- Preprocessing image: {0:>7}".format(n)
            sys.stdout.write(msg)
            sys.stdout.flush()

            if res != 1:
                failed_images.append(basename)
                continue
        
            new_filename = "{0}.jpg".format(basename)

            # Move the file from the tmp folder to the right grade folder.
            rename(join(tmp_path, new_filename),
                   join(data_dir, str(int(grade)), new_filename))

# Clean tmp folder.
rmtree(tmp_path)

if len(failed_images) != 0:
    print("Could not preprocess {} images.".format(len(failed_images)))
    print(", ".join(failed_images))

