FROM continuumio/anaconda3:2019.03

ENV bf /opt/babelfish
COPY . $bf
RUN conda update
RUN conda env update -n root -f  $bf/environment.yml

WORKDIR /Notebooks
#ENTRYPOINT ["python scripts/main.py"]
ENTRYPOINT ["jupyter lab"]


