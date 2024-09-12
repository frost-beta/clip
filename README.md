# Clip

Node.js module for the clip model.

Powered by [node-mlx](https://github.com/frost-beta/node-mlx), a machine
learning framework for Node.js.

Download model:

```console
huggingface download --to weights-clip \
                     --filter=*.json \
                     --filter=*.safetensors \
                     openai/clip-vit-large-patch14
```
