#!/usr/bin/bash
docker run \
    --mount type=bind,source=/Users/chrismalec/NumpyNN/models,target=/app/models \
    --mount type=bind,source=/Users/chrismalec/NumpyNN/configs,target=/app/configs \
    -d docker.io/numpynn:v1 \
    --epochs $1 \
    --batch_size $2 \
    --learning_rate $3 \
    --model_save_path $4 \
    --model_config $5