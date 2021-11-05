
FROM python:3.7



WORKDIR /app


ADD run.py /

COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade tensorflow
RUN python3 -m pip install lz4
RUN python3 -m pip install pandas
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install keras
RUN python3 -m pip install seaborn


COPY . .

CMD [ "python", "./run.py" ]
