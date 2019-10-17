FROM tensorflow/tensorflow:1.9.0-gpu-py3 AS memoria

COPY ./ivus /ivus

VOLUME /data /results

WORKDIR /ivus

RUN export LC_ALL=C.UTF-8 && \
    export LANG=C.UTF-8 && \
    export PYTHONHASHSEED=44 && \
    pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install -e .
