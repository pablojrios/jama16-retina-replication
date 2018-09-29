import pandas as pd
from os.path import join, splitext, basename, exists
import argparse
import numpy as np

common_images_file = "messidor_common.csv"

parser = argparse.ArgumentParser(description='Compare Messidor-1 and Messidor-2 datasets.')
parser.add_argument("--vendor_messidor_dir", help="Directory where Messidor annotations files reside.",
                    default="./vendor/messidor")

args = parser.parse_args()
vendor_messidor_dir = str(args.vendor_messidor_dir)

labels_messidor2 = join(vendor_messidor_dir, "messidor_data.csv")
labels_messidor1 = join(vendor_messidor_dir, "messidor-1_data.csv")

messidor1 = pd.read_csv(labels_messidor1)
messidor1['image_name'] = messidor1['image_name'].apply(lambda x: x.split(".")[0])
print(messidor1.head())

messidor2 = pd.read_csv(labels_messidor2)
messidor2['image_id'] = messidor2['image_id'].apply(lambda x: x.split(".")[0])
print(messidor2.head())
messidor2.to_csv(join(vendor_messidor_dir, "messidor_common.csv"), encoding='utf-8', index=False)

inner_join = pd.merge(messidor1, messidor2, how='inner', left_on=['image_name'], right_on=['image_id'])
print("Number of images in both datasets: {}".format(len(inner_join.index)))

inner_join.drop(['risk_of_macular_edema', 'image_id', 'adjudicated_dme', 'adjudicated_gradable'], axis=1, inplace=True)

inner_join['retinopathy_grade'] = inner_join['retinopathy_grade'].fillna(-1).astype(np.int64)
inner_join['adjudicated_dr_grade'] = inner_join['adjudicated_dr_grade'].fillna(-1).astype(np.int64)

inner_join.rename(columns={'retinopathy_grade': 'dr_grade_messidor-1', 'adjudicated_dr_grade': 'dr_grade_messidor-2'}, inplace=True)
inner_join.to_csv(join(vendor_messidor_dir, "messidor_common.csv"), encoding='utf-8', index=False)


