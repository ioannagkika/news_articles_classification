FROM python:3.10.1
WORKDIR /app
COPY label_encoder.pkl .
COPY requirements.txt .
COPY model-004.h5 .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]