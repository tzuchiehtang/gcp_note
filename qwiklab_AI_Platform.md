# AI Platform: Qwik Start

## Step 1: Get your training data

```bash
mkdir data
gsutil -m cp gs://cloud-samples-data/ml-engine/census/data/* data/

export TRAIN_DATA=$(pwd)/data/adult.data.csv
export EVAL_DATA=$(pwd)/data/adult.test.csv

```

## Step 2: Run a local training job
    > A local training job loads your Python training program and starts a training process in an environment that's similar to that of a live Cloud AI Platform cloud training job.

###     Step 2.1: Create files to hold the Python program

    > To do that, let's create three files. The first, called util.py, will contain utility methods for cleaning and preprocessing the data, as well as performing any feature engineering needed by transforming and normalizing the data.

## Step 3: Run your training job in the cloud

Now that you've validated your model by running it locally, you will now get practice training using Cloud AI Platform.

The initial job request will take several minutes to start, but subsequent jobs run more quickly. This enables quick iteration as you develop and validate your training job.

First, set the following variables:

```bash
%%bash
export PROJECT=$(gcloud config list project --format "value(core.project)")
echo "Your current GCP Project Name is: "${PROJECT}
> Your current GCP Project Name is: qwiklabs-gcp-00-2ef9c70664a9

%% .py | .ipynb

PROJECT = "qwiklabs-gcp-00-2ef9c70664a9"  # Replace with your project name
BUCKET_NAME=PROJECT+"-aiplatform"
REGION="us-central1"
os.environ["PROJECT"] = PROJECT
os.environ["BUCKET_NAME"] = BUCKET_NAME
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "2.1"
os.environ["PYTHONVERSION"] = "3.7"
```
2.


