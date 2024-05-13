FROM python:3.7.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get -y update
RUN apt-get install -y ffmpeg

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN useradd -m -u 1000 user

USER user

ARG HOME=/home/user

ENV HOME=${HOME} \
  PATH=${HOME}/.local/bin:$PATH \
  PYTHONPATH=${HOME}/app \
  PYTHONUNBUFFERED=1 \
  GRADIO_ALLOW_FLAGGING=never \
  GRADIO_NUM_PORTS=1 \
  GRADIO_SERVER_NAME=0.0.0.0 \
  GRADIO_THEME=huggingface \
  SYSTEM=spaces \
  HF_HOME=${HOME}/data/.huggingface


RUN mkdir -p ${HOME}/app
WORKDIR $HOME/app

COPY --chown=user ./*.py $HOME/app/

CMD ["python", "app.py"]

