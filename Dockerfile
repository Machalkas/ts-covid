FROM python:3.8-slim-buster AS builder
COPY . /app
WORKDIR /app
RUN pip install --user -r docker-reqs.txt

FROM python:3.8-slim-buster
COPY --from=builder /root/.local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/covidOracle.py /app/covidOracle.py
COPY --from=builder /app/data /app/data
COPY --from=builder /app/last_model.json /app/last_model.json
RUN apt-get update && apt-get install -y curl
WORKDIR /app
CMD [ "python", "main.py" ]