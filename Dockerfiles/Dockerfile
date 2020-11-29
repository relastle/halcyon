FROM tensorflow/tensorflow:1.14.0-gpu-py3

MAINTAINER Hiroki Konishi <relastle@gmail.com>

WORKDIR /opt
ADD requirements.txt /opt/requirements.txt
RUN pip install -r ./requirements.txt

ADD . /opt/halcyon
ENV PYTHONPATH=/opt/halcyon:${PYTHONPATH}

WORKDIR /opt/halcyon
ENTRYPOINT ["python3", "-m", "halcyon.console"]
