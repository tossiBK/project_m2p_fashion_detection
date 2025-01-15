import logging
from flask import Flask, request, jsonify
import mlflow.pyfunc
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import json
import os

# configure logging, we need some more informations in case something fail
# and we want to know what happen when the container startup and is ready
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # output to container logs
)

# load the model
logging.info("Loading model...")
model_path = "/app/model"
model = mlflow.pyfunc.load_model(model_path)
logging.info("Model loaded successfully.")

# load class labels from JSON file
labels_file = os.path.join(model_path, "class_to_idx.json")

if not os.path.exists(labels_file):
    logging.error(f"Labels file not found at {labels_file}")
    raise FileNotFoundError(f"Labels file not found at {labels_file}")

with open(labels_file, "r") as f:
    class_labels = json.load(f)

    # reverse mapping of class_labels to map indices back to labels
    # the index is saved in the opposite way, so we need this to get correct predictions
    # TODO maybe in future iteration better to save it already in the opposite assignment
    index_to_label = {v: k for k, v in class_labels.items()}

logging.info("Labels loaded")
logging.info(index_to_label)


# define Flask app
app = Flask(__name__)

# image preprocessing, this need to match the same pattern as we used in the training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint. We can use if we have issues with the API. Help to see if the container and flask is running"""
    logging.info("Health check ping received.")
    return jsonify({"status": "ok"}), 200

@app.route("/invocations", methods=["POST"])
def predict():
    """Handle prediction requests."""
    try:
        # log incoming request
        logging.info("Received prediction request.")
        data = request.json.get("instances", [])
        logging.info(f"Request data: {data}")

        if not data:
            logging.error("No input data provided.")
            return jsonify({"error": "No input data provided"}), 400

        # process each input
        predictions = []
        for instance in data:
            # convert the instance (list of pixel values) to a torch tensor
            # seems to be float32 to match what we send from the batch script
            tensor = torch.tensor(instance, dtype=torch.float32)  
            
            # check if the tensor has 3 dimensions, add a batch dimension if needed
            # we had errors not adding the batch dimentsion and the prediction failed with a numpy dimension error
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  
            
            # predict using the model
            preds = model.predict(tensor.numpy())  

            # get the index of the highest probability for the moment
            # in the future we should set a trashhold to evaluate images with no predictions at all (none matching)
            class_idx = np.argmax(preds[0])
            logging.info(f"Prediction: {class_idx}")
            
            # map index to class label
            predicted_class = index_to_label.get(class_idx, "Unknown")
            predictions.append(predicted_class)

        # log predictions for possible debugging
        # if the log get too much, we should set it only when using debug mode
        logging.info(f"Predictions: {predictions}")

        return jsonify(predictions)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# start the app and listen to all request on port 5001. 
# may be a config variable in the future if needed, but not at the moment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
