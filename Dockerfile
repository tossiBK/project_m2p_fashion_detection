# use the CUDA base image with Python 3.10 and Ubuntu 22.04
# matching our setup for using GPU
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# install dependencies
# image not have python etc, so we need to add it first
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# install pip packages (make sure to have a requirements.txt)
COPY model_artifacts/model/requirements.txt /app/requirements.txt

# install the Python dependencies from the requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# set up the application directory
WORKDIR /app

# copy the model directory into the container
COPY model_artifacts/model/ /app/model/
COPY model_artifacts/model_artifacts/ /app/model/

# copy the Flask app file into the container
# this will run our API
COPY app.py /app/

# expose the Flask app port
EXPOSE 5001

# set environment variables for Flask
ENV FLASK_ENV=production 
ENV FLASK_APP=app.py

# run the Flask app with gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app", "--access-logfile", "-", "--error-logfile", "-"]
