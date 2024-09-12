# Clip

Node.js module for the clip model.

Powered by [node-mlx](https://github.com/frost-beta/node-mlx), a machine
learning framework for Node.js.

## APIs

```typescript
import { core as mx } from '@frost-beta/mlx';

export interface ClipInput {
    labels?: string[];
    images?: Buffer[];
}

export interface ClipOutput {
    labelEmbeddings?: mx.array;
    imageEmbeddings?: mx.array;
}

export default class Clip {
    constructor(modelDir: string);
    computeEmbeddings({ labels, images }: ClipInput): Promise<ClipOutput>;
}
```

## License

MIT
