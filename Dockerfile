FROM ubuntu:latest
LABEL authors="Andy"

ENTRYPOINT ["top", "-b"]