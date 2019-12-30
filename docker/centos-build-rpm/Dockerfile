FROM centos:7
RUN yum install -y java-1.8.0-openjdk-devel which rpm-build redhat-rpm-config deltarpm git
RUN git clone https://github.com/konduitai/konduit-serving

WORKDIR konduit-serving
RUN ./mvnw clean package -Ppython,pmml,uberjar,tar,rpm -Dmaven.test.skip=true -Djavacpp.platform=linux-x86_64 -Dchip=cpu -Dspin.version=all

WORKDIR konduit-serving-rpm/target/rpm/konduit-serving-all-cpu/RPMS/x86_64/
RUN yum localinstall -y *.rpm && yum clean all 

RUN rm -r konduit-serving && rm -r ~/.m2