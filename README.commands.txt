Commands
========

Pre-processing:
--------------
./messidor2.sh --output_dir=./data/messidor2/bin
./messidor2.sh --output_dir=./data/messidor2/bin2.512 --large_diameter

Testing Ensambles:
-----------------
python evaluate.py -e --data_dir=./data/eyepacs/bin2.512/test/ --load_model_path=./tmp.7sept/model,./tmp.8sept/model
python evaluate.py -m --data_dir=./data/messidor/bin2.512/ --load_model_path=./tmp.7sept/model,./tmp.8sept/model

Training:
--------
python train.py --train_dir=./data/eyepacs/bin2.512/train/ --val_dir ./data/eyepacs/bin2.512/validation/ --large_diameter
python train.py --train_dir=./data/eyepacs/bin2/train/ --val_dir ./data/eyepacs/bin2/validation/

Testing:
-------
python evaluate.py -e --data_dir=./data/eyepacs/bin2.512/test/ --load_model_path=./tmp.18sept/model -so=./tmp.18sept/kaggle_test_op_pts.csv
python evaluate.py -m --data_dir=./data/messidor/bin2.512/ --load_model_path=./tmp.18sept/model -so=./tmp.18sept/messidor_test_op_pts.csv
python evaluate.py -m2 --data_dir=./data/messidor2/bin2.512/ --load_model_path=./tmp.18sept/model -so=./tmp.18sept/messidor-2_test_op_pts.csv