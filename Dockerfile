FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3


#SHELL ["/bin/bash", "--login", "-CMD"]

ARG USERNAME=daniel
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN echo "Set disable_coredump false" >> /etc/sudo.conf
RUN sudo apt-get --assume-yes install libosmesa6-dev
RUN sudo apt-get --assume-yes install libc-dev
RUN sudo pip install patchelf
RUN sudo apt-get --assume-yes install htop
# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
WORKDIR /home/$USERNAME/COMPER-RELEASE-CODE
COPY . .
RUN 
RUN sudo chown -R $USERNAME /home/$USERNAME/COMPER-RELEASE-CODE
RUN pip install faiss-gpu
RUN pip install click=8.0.1
RUN pip install opencv=3.4.2
RUN pip install pandas=1.2.5
RUN pip install pillow=8.3.1
RUN pip install scikit-learn
