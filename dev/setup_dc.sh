#!/bin/bash

root_dir=$(dirname $(dirname $(realpath $0)))
export ROOT_DIR=${root_dir}
export USERNAME=$(whoami)

IMAGE_NAME_1="registry-intl.cn-hongkong.aliyuncs.com/flavius/chatbot-dev:latest"
IMAGE_NAME_2="markify-service:latest"
CONTAINER_NAME_1="${USERNAME}_chatbot_python"
CONTAINER_NAME_2="${USERNAME}_markify_service"
PROJECT_NAME="${USERNAME}_project"

start_docker_compose() {
    # remove old images if exists
    docker image rmi ${IMAGE_NAME_1} 2>/dev/null || true
    docker image rmi ${IMAGE_NAME_2} 2>/dev/null || true

    # start docker-compose
    echo "Starting docker-compose with project name: ${PROJECT_NAME}..."
    docker compose -p ${PROJECT_NAME} up -d --build
    if [ $? -ne 0 ]; then
        echo "docker compose failed to start. Please check the logs for more information."
        exit 1
    fi
    echo "docker compose started successfully."

    # print container names
    echo "You may run \"docker exec -it ${CONTAINER_NAME_1} /bin/bash\" to enter the main app container"
    echo "You may run \"docker exec -it ${CONTAINER_NAME_2} /bin/bash\" to enter the MinerU service container"
}

set -e
### on host:
start_docker_compose