FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN python -m venv venv \
    && . venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "stream:app", "--host", "0.0.0.0", "--port", "8000"]