# News Articles Classification

This folder contains distilbert fine-tuned for classifying news articles. The model has been fine-tuned using the News Category Dataset dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

The code of the report was developed in Google Colab.

You can download the the weights of the pre-trained model (filename: model-004.h5) and the label encoder (filename:label_encoder.pkl) from [Google Drive](https://drive.google.com/drive/folders/1B1r0_WMrFG7YUZNkghmABq2kqxS3_I3o?usp=sharing). 

# Dockerized Model

In the dockerized_model folder you will find the dockerized version of the news articles classifier model that serves an HTTP API with Flask.

## Instructions

1. Clone the repo locally or just download the current folder.

2. Download the the weighs of the pre-trained model (filename: model-004.h5) and the label encoder (filename:label_encoder.pkl) from [Google Drive](https://drive.google.com/drive/folders/1B1r0_WMrFG7YUZNkghmABq2kqxS3_I3o?usp=sharing) and place them into the same folder (dockerized_model).

3. Get into the folder, open a terminal and run the command

```bash
docker build -t dockerized_model .
```

This may take a few minutes.

4. When the building is completed run the command

```bash
docker run -p 5000:5000 dockerized_model
```

Now the container is up and running and our model is ready to receive requests

5. Open an new terminal and send a curl request. An example is the following

```bash
curl -X POST http://0.0.0.0:5000/predict -H 'Content-Type: application/json' -d '{"text": "Messi did not score today"}'
```

Note that the model receives requests in a json format with the structure: 
{"text": "Messi did not score today"}, where text is the joined headline and content of an article
and outputs messages in a json format with the structure:
{"label": "SPORTS", "confidence": 0.99, "version": "0.0.1"}
where the predicted_label is a string containing the predicted by the model label, the confidence_of_prediction is a float containing the confidence of the prediction of the model and "version" is a string containing the version of the model (currently 0.0.1)
