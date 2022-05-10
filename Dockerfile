FROM registry.fedoraproject.org/fedora:36
RUN dnf install -y python3-pip && dnf clean all
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY 11bot.py /usr/local/bin
ENTRYPOINT ["/usr/local/bin/11bot.py"]
