Commands
========

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
python evaluate.py -e --data_dir=./data/eyepacs/bin2.512/test/

python evaluate.py -m --data_dir=./data/messidor/bin2.512/
