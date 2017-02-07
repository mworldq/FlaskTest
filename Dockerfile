FROM tensorflow/tensorflow

#Maintainer
MAINTAINER Caocao <martin.mengdj@gmail.com>

RUN \
 DEBIAN_FRONTEND=noninteractive apt-get update && \
 DEBIAN_FRONTEND=noninteractive apt-get -y install lrzsz unzip

ADD ./ /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["/usr/bin/python","app.py&"]
