FROM python:3

ENV IMG_PATH=/usr/src/images
RUN mkdir -p ${IMG_PATH}

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .