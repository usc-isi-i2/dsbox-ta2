FROM registry.datadrivendiscovery.org/jpl/docker_images/jpl_base_python3.5
MAINTAINER D3M


RUN \
pip3 install punk && \
pip3 install langdetect && \
pip3 install zerorpc && \
pip3 install stopit

EXPOSE 8888

#COPY start.sh /home/start.sh
#RUN set -ex \ && chmod 755 /home/start.sh
#WORKDIR /home
#CMD ["./start.sh"]

# Copy the TA2 code to the docker container
ENV CODE /home
COPY python $CODE

# work-around for non-interactive shells
RUN echo '#!/bin/bash\npython3 $CODE/ta2_search "$@"' > /usr/bin/ta2_search
RUN chmod +x /usr/bin/ta2_search
