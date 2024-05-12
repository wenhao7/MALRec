FROM continuumio/miniconda

COPY . /Recommendation/

WORKDIR Recommendation

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "MalRec", "/bin/bash", "-c"]

RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

ENV PYTHONPATH=/Recommendation/app

COPY app.py .
ENTRYPOINT ["conda", "run", "-n", "MalRec", "python", "app.py"]