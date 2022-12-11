# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster
LABEL maintainer "Ivan Skvortsov <iaskvortsov@edu.hse.ru>"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install "poetry==1.2.2" && poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml /app/
COPY . ./

RUN poetry install

CMD [ "poetry", "run" , "flask", "run", "--host=0.0.0.0"]