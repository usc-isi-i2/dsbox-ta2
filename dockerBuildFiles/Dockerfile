FROM registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9-20200212-063959

RUN mkdir -p /user_opt
RUN mkdir -p /output
RUN mkdir -p /input

ENV CODE /user_opt/dsbox
ENV TA2 $CODE/dsbox-ta2/python

RUN pip3 install --upgrade pip

RUN mkdir $CODE

RUN cd $CODE && ls -l

# Install TA3TA2-API

RUN cd $CODE && pip3 install -e git+https://gitlab.com/datadrivendiscovery/ta3ta2-api.git@1214abaac7cccd3f578e9589509b279bd820a758#egg=ta3ta2-api


# Install TA3TA2-API protocol buffers definition for convenience
RUN git clone https://gitlab.com/datadrivendiscovery/ta3ta2-api.git ta3ta2-api-proto && cd ta3ta2-api-proto && git checkout b65b0d28be4b3aa1de84e695c7c10fcc9f4cb584

# Install dummy_ta3
RUN cd $CODE && pip3 install -e git+https://gitlab.com/datadrivendiscovery/dummy-ta3.git@0a82119bc90c7b41b5bf0a3d1c00fe0ed12e9b91#egg=dummy_ta3

# Install extract packages for development
RUN apt-get update
RUN apt-get -y install emacs

RUN cd $CODE \
&& git clone https://github.com/usc-isi-i2/dsbox-ta2.git \
&& cd dsbox-ta2 \
&& git checkout 0e80dfcec970846966a50e075ee3b5892340f1c9 \
&& pip install -r requirements.txt


COPY d3mStart.sh /user_opt
COPY client.sh /user_opt
COPY score-pipeline.sh /user_opt

RUN chmod a+x /user_opt/d3mStart.sh /user_opt/client.sh /user_opt/score-pipeline.sh

RUN pip3 list


CMD ["/user_opt/d3mStart.sh"]
