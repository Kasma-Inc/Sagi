#!/bin/bash

# Set default values for the directories if not provided
if [ -z "$SAGI_DIR" ]; then
    echo "SAGI_DIR is not set, using default path $(pwd)"
    SAGI_DIR="$(pwd)"
    export SAGI_DIR=${SAGI_DIR}
fi

if [ -z "$DOCKER_SOCKET_PATH" ]; then
    echo "DOCKER_SOCKET_PATH is not set, using default path /var/run/docker.sock"
    DOCKER_SOCKET_PATH="/var/run/docker.sock"
    export DOCKER_SOCKET_PATH=${DOCKER_SOCKET_PATH}
fi

export USERNAME=$(whoami)
CONTAINER_NAME="${USERNAME}_sagi-dev"
COMPOSE_PROJECT_NAME="${USERNAME}_sagi-dev-dc"

start_docker_compose() {
    # stop container if it is running
    docker stop ${CONTAINER_NAME} 2>/dev/null || true

    # remove container
    docker rm ${CONTAINER_NAME} 2>/dev/null || true

    # start docker-compose
    echo "Starting docker-compose with project name: ${COMPOSE_PROJECT_NAME}..."
    docker compose -f docker-compose.yml -p ${COMPOSE_PROJECT_NAME} up -d --build
    if [ $? -ne 0 ]; then
        echo "docker-compose failed to start. Please check the logs for more information."
        exit 1
    fi
    echo "docker-compose started successfully."

    # print container name
    echo "You may run \"docker exec -it ${CONTAINER_NAME} /bin/bash\" to enter the sagi container"
}

set -e
### on host:
start_docker_compose