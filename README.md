# DL-Based Solution Proposal for Monitoring Concreting Operations

## Group 8
Team members:
* Aïcha Bokbot 
* GAN Jiayu
* LI Delong
* YU Honghao
* ZHAO Qiwen

## Clone this repo with credential of azure as submodule
`git clone --recurse-submodule {URL}`

If you forgot to clone with submodule: `git submodule update --init --recursive`


## Data 
1. Detect_Train_Set, Detect_Train_Set_Bis and Analytic_Train_Set were distributed via azure cloud storage.

   The pipeline will automatically fetch the images and annotations from remote server.
   
2. Revised polygon annotations for Analytic_Train_Set was distributed via direct file transfer. 

   You **must upload the corresponding files** (files in poly.tar) to IMAGE_DIR directory and **rename** the poly_json files as

   **f"{sample_id}_poly.json"**

3. Detect_Test_Set was distributed via direct file transfer. 

   You **must upload the corresponding files** (images in Detection_Test_Set_Img.tar) to INFERENCE_DIR.

# Install packages
Install required python packages by executing
`pip install -r requirements.txt`


# Getting Started
* [1 - Explore_data_set](notebooks/1 - Explore_data_set.ipynb) is for downloading data from azure server and visualize some randomly-picked samples. Executing this notebook is a prerequisite for the following notebooks.

* (OPTIONAL.) Launch model training by 
  `python3 Train.py --steps=400 --head_epochs=0 --all_epochs=100 --lr=0.00002 --init=last --branch=chronsite20201003T1222 --cp=77`

* [2 - Evaluate_maskrcnn](notebooks/2 - Evaluate_maskrcnn.ipynb) is for evaluating the trained model on validation set. Certain 
figures are to be generated in this notebook for presentation, such as mAP and P-R curve.

* [3 - Analytic_model_heatmap](notebooks/3 - Analytic_model_heatmap.ipynb) is to generate heatmap of concentration of workers on the field.

* [4 - Inference](notebooks/4 - Inference.ipynb) is to retrieve predictions from our trained model on the test set, to visualize the predictions and to assemble the output into an integrated json file in coco format.

* [5 - Analysis_model](Analysis_model.ipynb) is to generate DataFrame used for our analysis model, includes construction area calculation, current status detection, etc.