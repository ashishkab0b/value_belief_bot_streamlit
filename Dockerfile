
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && apt-get clean

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENV ENV_TYPE=production

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -t streamlit . && docker run -p 80:8501 streamlit