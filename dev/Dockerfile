FROM node:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y neovim git less cargo python3-dev

# set git autocompletion & set safe directory &  set bashrc
RUN echo "source /usr/share/bash-completion/completions/git" >>~/.bashrc && git config --global --add safe.directory /chatbot && \
    cat <<'EOF' >>~/.bashrc
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac
if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt
EOF

# install pnpm for node package management, mcp server stdio required
RUN wget -qO- https://get.pnpm.io/install.sh | ENV="$HOME/.bashrc" SHELL="$(which bash)" bash - && \
    export PNPM_HOME="/root/.local/share/pnpm" && \
    export PATH="$PNPM_HOME:$PATH" && \
    echo 'export PNPM_HOME="/root/.local/share/pnpm"' >> ~/.bashrc && \
    echo 'export PATH="$PNPM_HOME:$PATH"' >> ~/.bashrc && \
    ln -s /root/.local/share/pnpm/pnpm /usr/local/bin/pnpm

# install uv for python package management, mcp server stdio required
RUN wget -qO- https://astral.sh/uv/install.sh | sh && \
    source "${HOME}/.local/bin/env" && \
    uv venv "${HOME}/.venv" --python python3.12 && \
    source ${HOME}/.venv/bin/activate && uv pip install pip && \
    cat <<'EOF' >>~/.bashrc
if [ -z "$VIRTUAL_ENV" ]; then
    source ${HOME}/.venv/bin/activate
fi
EOF

WORKDIR /chatbot
