FROM nvidia/cuda:10.2-base
# Set locale
ENV LANG=C.UTF-8
# Avoid apt installations to require user interaction
ENV DEBIAN_FRONTEND=noninteractive

# Install pyenv
## Install dependencies for pyenv
RUN apt update && apt install --no-install-recommends -y \
    git curl unzip git make build-essential libssl-dev python3-pip python3-setuptools\
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl zlib1g-dev
## Install pyenv from online script
RUN curl https://pyenv.run | bash
## Add bashrc lines to configure pyenv
RUN echo 'export PATH="~/.pyenv/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
## Configure the current shell
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:${PATH}"
RUN eval $(pyenv init --path)

# Switch to the /app workdir
WORKDIR /app

# Install repository Python version and set-up virtual environment
COPY .python-version ./
RUN pyenv install

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
RUN pip install --upgrade pip
COPY poetry.lock  pyproject.toml ./
RUN poetry install

# Copy the repository
COPY .git .git
COPY src src
COPY main.py ./
COPY batch.sh ./
RUN chmod 755 /app/batch.sh

ENTRYPOINT ["/app/batch.sh"]