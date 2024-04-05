# You can build from any valid docker image, we use cuda here as an example as your model likely
# needs the cuda libraries to utilise a GPU
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Install python and any requirements. The example uses gdal to process the provided raster
RUN apt-get update &&  \
    apt-get install -y python3-pip gdal-bin && \
    apt-get clean # clean up apt cache after install to keep image size small

RUN mkdir -p /opt/mymodel

# COPY and install requirements separately to speed up future builds
COPY requirements.txt /opt/mymodel/
RUN pip install -r /opt/mymodel/requirements.txt

COPY src/ /opt/mymodel/src
WORKDIR /opt/mymodel/src/

ENTRYPOINT ["python3", "main.py"]