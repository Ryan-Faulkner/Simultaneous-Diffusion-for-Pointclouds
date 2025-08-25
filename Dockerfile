# HDVNet and Randlanet docker
FROM pytorch/pytorch:latest
RUN apt-get update && apt-get -y install git && git config --global http.sslverify "false"

COPY requirements.txt /tmp/

RUN apt-get -y install build-essential
RUN apt-get -y install gcc
RUN pip install --upgrade --no-cache-dir -r /tmp/requirements.txt
RUN pip install --index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple mkl_fft
