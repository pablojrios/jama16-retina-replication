Comandos nuevos datasets
========================

Pre-processing:
--------------
# bin300: con 50K imágenes clase 0 de training de 300 pixels
./eyepacs.sh --redistribute --pool_dir=./data/eyepacs/pool/ --output_dir=./data/eyepacs/bin300/
python train.py --train_dir=./data/eyepacs/bin300/train/ --val_dir=./data/eyepacs/bin300/validation/
./messidor2.sh --output_dir=./data/messidor2/bin


Commands
========

Pre-processing:
--------------
./messidor2.sh --output_dir=./data/messidor2/bin
./messidor2.sh --output_dir=./data/messidor2/bin2.512 --large_diameter
./messidor2.sh --output_dir=./data/messidor2/bin2.512.v2/ --large_diameter
./eyepacs.sh --redistribute --pool_dir=./data/eyepacs/pool512/ --seed=12345 --large_diameter --output_dir=./data/eyepacs/bin2.22sept/
./eyepacs.sh --redistribute --pool_dir=./data/eyepacs/pool512/ --large_diameter --output_dir=./data/eyepacs/bin2.512.filename/
    Finding images...
    Found 65337 images in class 0.
    Found 6203 images in class 1.
    Found 13151 images in class 2.
    Found 2087 images in class 3.
    Found 1914 images in class 4.
    Creating directories for data sets
    Gathering 60000 images for train set (0/2)
    Gathering 16458 images for train set (1/2)
    Gathering 8096 images for test set (0/2)
    Gathering 694 images for test set (1/2)
    >> Converting image 61167/61167 shard 7
    >> Converting image 15291/15291 shard 7
    >> Converting image 8790/8790 shard 3
./eyepacs.sh --redistribute --pool_dir=./data/eyepacs/pool512/ --large_diameter --output_dir=./data/eyepacs/bin2.512.gradonly/


Testing Ensambles:
-----------------
python evaluate.py -e --data_dir=./data/eyepacs/bin2.512/test/ --load_model_path=./tmp.7sept/model,./tmp.8sept/model
python evaluate.py -m --data_dir=./data/messidor/bin2.512/ --load_model_path=./tmp.7sept/model,./tmp.8sept/model

Training:
--------
python train.py --train_dir=./data/eyepacs/bin2.512/train/ --val_dir=./data/eyepacs/bin2.512/validation/ --large_diameter
python train.py --train_dir=./data/eyepacs/bin2/train/ --val_dir=./data/eyepacs/bin2/validation/
python train.py --train_dir=./data/eyepacs/bin2/train/ --val_dir=./data/eyepacs/bin2/validation/ --optimizer=adam

Testing:
-------
python evaluate.py -e --data_dir=./data/eyepacs/bin2.512/test/ --load_model_path=./tmp.18sept/model -so=./tmp.18sept/kaggle_test_op_pts.csv
python evaluate.py -m --data_dir=./data/messidor/bin2.512/ --load_model_path=./tmp.18sept/model -so=./tmp.18sept/messidor_test_op_pts.csv
python evaluate.py -m2 --data_dir=./data/messidor2/bin2.512/ --load_model_path=./tmp.18sept/model -so=./tmp.18sept/messidor-2_test_op_pts.csv

python evaluate.py -e --data_dir=./data/eyepacs/bin2.512.fileid/test/ --load_model_path=./tmp/model -so=./tmp/kaggle_test_op_pts.csv -p=./tmp/kaggle_test_predictions.csv
python evaluate.py -m2 --data_dir=./data/messidor2/bin2.512.fileid/ --load_model_path=./tmp/model -so=./tmp/messidor2_test_op_pts.csv -p=./tmp/messidor2_test_predictions.csv -op=0.1658
