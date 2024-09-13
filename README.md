# Clip

Node.js module for the [CLIP model](https://openai.com/index/clip/).

Powered by [node-mlx](https://github.com/frost-beta/node-mlx), a machine
learning framework for Node.js.

## APIs

```typescript
import { core as mx } from '@frost-beta/mlx';

export interface ClipInput {
    labels?: string[];
    images?: BufferType[];
}

export interface ClipOutput {
    labelEmbeddings?: mx.array;
    imageEmbeddings?: mx.array;
}

export class Clip {
    constructor(modelDir: string);
    computeEmbeddings({ labels, images }: ClipInput): Promise<ClipOutput>;
    static computeCosineSimilaritiy(a1: mx.array, a2: mx.array): number;
    static computeCosineSimilarities(x1: mx.array, x2: mx.array): number[];
}
```

## License

MIT
