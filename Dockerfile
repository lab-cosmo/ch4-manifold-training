# syntax=docker/dockerfile:1

FROM pytorch/pytorch

WORKDIR /app


RUN apt-get update
RUN apt-get install -y g++
RUN apt-get install -y git
RUN apt-get install -y apt-utils
RUN apt-get install -y cmake
RUN apt-get install -y clang
RUN pip3 install numpy scipy ase


RUN git clone https://github.com/lab-cosmo/librascal.git
WORKDIR /app/librascal
RUN git checkout feat/gaptools
# https://github.com/lab-cosmo/librascal/releases/tag/cnn-counterexample
RUN git checkout 3fd08e4bdb70060ad797ffc63fb591858bb62661
RUN pip3 install .

WORKDIR /app

RUN pip3 install jupyter
COPY . .

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

