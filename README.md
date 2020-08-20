# Matroid_Asst


# Quickstart

1. create the conda environment provided by using  `conda env create -f environment.yml`

2. Activate conda environment via `conda activate Matroid`

3. Train and evaluate the model via `python train_evaluate_model.py` 

(**NOTE** You can access optional params (`num_epochs`, `save_model_dir`, etc) by running `python train_evaluate_model.py -h`)

Final  trained model is contained in the `/descriptorGenderClassifier` directory

## OPTIONAL REBUILD OF FEATURE VECTORS

(**NOTE** Feature vectors are already provided in the `/data` folder, you can skip the next two steps)

1. Download the dataset [here](https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz.) and unzip it into `$DATASET_DIR` you would prefer.


2. Rebuild the feature vectors used for our final model by deleting `/data/female_descriptors.npy` and `/data/male_descriptors.npy`. Then run `python create_feature_vectors.py --data_dir $DATASET_DIR` 

3. You can then rerun `python train_evaluate_model.py --data_dir $DATASET_DIR` 

# Results: 

After running `python train_evaluate_model.py --num_epochs 10 --batch_size 16 --train_prop 0.70`

classification accuracy overall on Test set: 0.956048387097

classification metrics per class on Test set: 

              precision    recall  f1-score   support

        male       0.97      0.94      0.96      2513
      female       0.94      0.97      0.96      2447

   micro avg       0.96      0.96      0.96      4960
   macro avg       0.96      0.96      0.96      4960
weighted avg       0.96      0.96      0.96      4960


# Citations

Installed vggface model from [keras-vggface](https://github.com/rcmalli/keras-vggface) by Refik Can Malli 

# Discussion

Before building my model, I preprocess the vggface descriptors for male and female faces into `.npy` files via the `create_feature_vectors.py` script. 

Afterward, I use those preprocessed feature vectors as the inputs to a small feed-forward fully connected model in the `train_evaluate_model.py` script. Full description of the gender classifier model is contained in the script. Since the feature vectors provided by 
