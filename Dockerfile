FROM ubuntu:latest
MAINTAINER Rahul Ghosh "ghrahul2016@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
ADD . /flask-app/app.py
ADD . /flask-app
WORKDIR /flask-app
RUN pip install Flask==1.0.2
ENTRYPOINT ["python"]
CMD ["app.py"]