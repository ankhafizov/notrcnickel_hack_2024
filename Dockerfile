FROM python:3.11-slim

RUN pip3 install torch==2.0.1 torchvision==0.15.2
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx && rm -rf /var/lib/apt/lists/* 


WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8888", "--server.address=0.0.0.0"]
