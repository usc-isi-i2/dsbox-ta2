FROM registry.datadrivendiscovery.org/jpl/docker_images/jpl_base_python2.7
MAINTAINER D3M


RUN \
pip install corexcontinuous && \
pip install featuretools && \
pip install langdetect && \
pip install zerorpc && \
pip install stopit

EXPOSE 8888

#COPY start.sh /home/start.sh
#RUN set -ex \ && chmod 755 /home/start.sh
#WORKDIR /home
#CMD ["./start.sh"]

# Copy the TA2 code to the docker container 
ENV CODE /home
COPY python $CODE
 
# work-around for non-interactive shells
RUN echo '#!/bin/bash\npython $CODE/main.py search "$@"' > /usr/bin/ta2_search
RUN chmod +x /usr/bin/ta2_search
