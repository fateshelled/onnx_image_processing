ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
