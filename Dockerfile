FROM  nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         wget \
         cmake \
         ca-certificates \
         git \
         curl \
         python3.6-dev \
         gcc \
         python3-setuptools \
         python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN python3 -m pip install git+https://github.com/allenai/allennlp.git

# Copy select files needed for installing requirements.
# We only copy what we need here so small changes to the repository does not trigger re-installation of the requirements.

COPY ./reading_comprehension ./reading_comprehension
COPY run_model.sh .
COPY predictor.py .

# EXPOSE 8000
Cmd ["/bin/bash"]
