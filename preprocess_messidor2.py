import argparse
import csv
import sys
from shutil import rmtree
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from lib.preprocess import scale_normalize

parser = argparse.ArgumentParser(description='Preprocess Messidor-2 data set.')
parser.add_argument("--data_dir", help="Directory where Messidor-2 resides.",
                    default="data/messidor2")
parser.add_argument("--large_diameter", action="store_true",
                    help="diameter of fundus to 512 pixels.")

args = parser.parse_args()
data_dir = str(args.data_dir)
large_diameter = bool(args.large_diameter)

labels = join(data_dir, 'messidor-2_data.csv')

# Create directories for grades.
[makedirs(join(data_dir, str(i))) for i in [0, 1, 2, 3, 4]
 if not exists(join(data_dir, str(i)))]

# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []
reflejos_images = []

diameter = 512 if large_diameter else 299
print("Large fundus diameter={}".format(large_diameter))

with open(labels, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)

    for i, row in enumerate(reader):
        basename, grade, dme, gradable, reflejos = row

        # Skip non-gradable images, son solo 4 imagenes:
        # 20060411_58550_0200_PP.png,,,0
        # IM002385.jpg,,,0
        # IM003718.jpg,,,0
        # IM004176.jpg,,,0
        if gradable == '0':
            failed_images.append(basename)
            continue
        elif reflejos == '1':
            reflejos_images.append(basename)
            continue

        im_path = join(data_dir, "IMAGESv2/{}".format(basename))

        # Find contour of eye fundus in image, and scale
        #  diameter of fundus to 299 pixels and crop the edges.
        res = scale_normalize(save_path=tmp_path,
                              image_path=im_path,
                              diameter=diameter, verbosity=0)

        # Status message.
        msg = "\r- Preprocessing image: {0:>6}".format(i+1)
        sys.stdout.write(msg)
        sys.stdout.flush()

        if res != 1:
            failed_images.append(basename)
            continue

        new_filename = "{0}.jpg".format(basename.split(".")[0])

        # Move the file from the tmp folder to the right grade folder.
        rename(join(tmp_path, new_filename),
               join(data_dir, str(int(grade)), new_filename))

# Clean tmp folder.
rmtree(tmp_path)

if len(failed_images) != 0:
    print("\nCould not preprocess {} images.".format(len(failed_images)))
    print(", ".join(failed_images))

if len(reflejos_images) != 0:
    print("\nDid not preprocess {} images because of reflexions.".format(len(reflejos_images)))
    print(", ".join(reflejos_images))
