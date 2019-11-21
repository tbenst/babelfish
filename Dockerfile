# 10.1-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda@sha256:d425e7b2f999c497c51d2289b938c3b1aa125bdc81128d3811635d11a641afc0

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

COPY strict_environment /tmp/
RUN conda create --name babel --file /tmp/strict_environment
RUN echo ". activate babel" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
WORKDIR /Notebooks

CMD ["/opt/conda/envs/babel/bin/jupyter lab --ip 0.0.0.0 --no-browser --port 8880 --allow-root"]
