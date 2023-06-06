FROM gcr.io/deeplearning-platform-release/pytorch-gpu

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y screen vim tmux zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN sed -i 's/robbyrussell/avit/g' ~/.zshrc
RUN echo "alias lss='ls -rtlh'" >> ~/.zshrc

COPY . /root/time_interpret
WORKDIR /root/time_interpret
RUN pip install --no-deps -e .