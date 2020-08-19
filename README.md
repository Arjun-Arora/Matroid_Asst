# Matroid_Asst

#Quickstart

1. create the conda environment provided by using  **TODO**

2. download the caffe model from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/). Unzip it in the root directory.

3. use the following command to actually create the tensorflow model from the caffe model (*Note* this command specifies you use the root directory's `.prototxt` **not** the one provided by the download link above. This is important for the convertor since the original caffe model uses an older `.prototxt` syntax which needed to be modified to be used with modern caffe and tensorflow.

`./caffe-tensorflow/convert.py --caffemodel ./vgg_face_caffe/VGG_FACE.caffemodel --data-output-path vggface.npy --code-output-path vggface.py ./VGG_FACE_deploy.prototxt `

4. Download the dataset [here](https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz.) and unzip it into to whatever `$DATASET_DIR` you would prefer.

5. Build the feature vectors used for our final model by running `python create_feature_vectors.py --data_dir $DATASET_DIR`
