FROM ubuntu:20.04
MAINTAINER Natalie Prange prange@cs.uni-freiburg.de
WORKDIR /home/
RUN apt-get update
RUN apt-get install -y python3 python3-pip wget vim curl bc
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_lg
COPY src src
COPY scripts scripts
COPY Makefile .
COPY README.md .
# Enable Makefile target autocompletion
RUN echo "complete -W \"\`grep -oE '^[a-zA-Z0-9_.-]+:([^=]|$)' ?akefile | sed 's/[^a-zA-Z0-9_.-]*$//'\`\" make" >> ~/.bashrc
# Add src directory to the PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:src"
# Files created in the docker container should be easily accessible from the outside
CMD umask 000; /bin/bash;

# Build the container:
# docker build -t natural-entity-types .

# Run the container:
# docker run -it -v $(pwd)/data/:/home/data -v $(pwd)/models/:/home/models -v $(pwd)/benchmarks/:/home/benchmarks natural-entity-types
