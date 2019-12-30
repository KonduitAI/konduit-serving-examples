FROM ubuntu:latest 

RUN apt-get update && apt-get install openjdk-8-jdk git --no-install-recommends -y && \
git clone https://github.com/konduitai/konduit-serving

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

WORKDIR konduit-serving
RUN ./mvnw clean package -Ppython,pmml,uberjar,tar,deb -Dmaven.test.skip=true -Djavacpp.platform=linux-x86_64 -Dchip=cpu -Dspin.version=all && \
apt-get install ./konduit-serving-deb/target/konduit-serving-all-cpu_0.1.0-SNAPSHOT.deb &&\ 
apt-get clean

RUN rm -r konduit-serving && rm -r ~/.m2