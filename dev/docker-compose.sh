#!/bin/bash

# 1) Create the plug‑in directory (if it does not exist)
mkdir -p ~/.docker/cli-plugins

# 2) Download the binary that matches your OS/CPU
curl -SL \
  "https://github.com/docker/compose/releases/download/v2.35.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o ~/.docker/cli-plugins/docker-compose

# 3) Make it executable
chmod +x ~/.docker/cli-plugins/docker-compose

# 4) Verify your installation
docker compose version