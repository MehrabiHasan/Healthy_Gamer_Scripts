#Download base image ubuntu 22.04
FROM nvcr.io/nvidia/pytorch:23.04-py3 as builder-image 

ARG DEBIAN_FRONTEND=noninteractive
# Add the deadsnakes PPA for Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common libgl1-mesa-glx cmake protobuf-compiler && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

#common labels 
LABEL maintainer = "mhasan11@gmu.edu"
LABEL version = "0.1"
LABEL description = "Custom DockerFile for Healthy Gamer Scripts"

# avoid stuck build due to user prompt


RUN apt-get update && apt-get install --no-install-recommends -y python3.10 python3.10-dev  python3.10-venv python3-pip python3-wheel build-essential git && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python3.10 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"


# install requirements
COPY requirements.txt .

RUN pip3 install git+https://github.com/guillaumekln/faster-whisper
RUN pip3 install git+https://github.com//yt-dlp/yt-dlp 
RUN pip3 install --no-cache-dir -r requirements.txt 

FROM nvcr.io/nvidia/pytorch:23.04-py3 AS runner-image
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common libgl1-mesa-glx cmake protobuf-compiler && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get update && apt-get install --no-install-recommends -y python3.10 python3.10-venv && \
    apt-get install ffmpeg -y && \ 
	apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home myuser
COPY --from=builder-image /home/myuser/venv /home/myuser/venv

USER myuser
RUN mkdir /home/myuser/code /home/myuser/code/src/ /home/myuser/code/src/scripts /home/myuser/code/src/data
WORKDIR /home/myuser/code
COPY ./src/api /home/myuser/code/src/api
COPY ./src/schemas /home/myuser/code/src/schemas

EXPOSE 5000

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"