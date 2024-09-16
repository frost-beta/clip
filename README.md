# Clip

Node.js module for the [CLIP model](https://openai.com/index/clip/).

Powered by [node-mlx](https://github.com/frost-beta/node-mlx), a machine
learning framework for Node.js.

## APIs

```typescript
import { core as mx } from '@frost-beta/mlx';

export type ImageInputType = Buffer | ArrayBuffer | string;

export interface ProcessedImage {
    data: Buffer;
    info: sharp.OutputInfo;
}

export interface ClipInput {
    labels?: string[];
    images?: ProcessedImage[];
}

export interface ClipOutput {
    labelEmbeddings?: mx.array;
    imageEmbeddings?: mx.array;
}

export class Clip {
    constructor(modelDir: string);
    processImages(images: ImageInputType[]): Promise<ProcessedImage[]>;
    computeEmbeddings({ labels, images }: ClipInput): ClipOutput;
    /**
     * Compute the cosine similarity between 2 embeddings.
     */
    static computeCosineSimilaritiy(a1: mx.array, a2: mx.array): mx.array;
    /**
     * Compute the cosine similarities between 2 arrays of embeddings.
     *
     * A tuple will be returned, with the first element being the cosine
     * similarity scores, and the second element being the indices sorted by
     * their scores from larger to smalller.
     */
    static computeCosineSimilarities(x1: mx.array | number[], x2: mx.array | number[]): [mx.array, mx.array];
}
```

## License

MIT
