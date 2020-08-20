# Matroid_Asst

#Quickstart

1. create the conda environment provided by using  **TODO**

2. install vggface model from rcmalli via
	- `pip install git+https://github.com/rcmalli/keras-vggface.git`
	- `pip install keras_vggface`

3. Download the dataset [here](https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz.) and unzip it into to whatever `$DATASET_DIR` you would prefer.

4. Build the feature vectors used for our final model by running `python create_feature_vectors.py --data_dir $DATASET_DIR`
