FROM python:3.10

COPY . .

# FROM [path to files from the folder we run docker run]
# TO [current WORKDIR]
# We copy our files (files from .dockerignore are ignored)
# to the WORKDIR

RUN pip install --no-cache-dir -r requirements.txt

# OK, now we pip install our requirements

EXPOSE 8000

# Instruction informs Docker that the container listens on port 80

WORKDIR .


CMD uvicorn stream:app --host 0.0.0.0 --port 8001
