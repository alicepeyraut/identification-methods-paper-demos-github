
FROM ghcr.io/alicepeyraut/identification-methods-paper-demos-github:latest

ARG NB_USER=jovyan
ARG NB_UID=1000
USER ${NB_USER}
ENV HOME /home/${NB_USER}
COPY --chown=${NB_UID} . ${HOME}
