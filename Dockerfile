FROM python:3.8.2

RUN mkdir /workspace
WORKDIR /workspace

COPY ./requirements.txt /workspace/
COPY ./models /workspace/models
COPY ./network.py /workspace/
COPY ./network_failures.py /workspace/

RUN pip install -r /workspace/requirements.txt
