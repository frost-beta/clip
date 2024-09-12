import {core as mx} from '@frost-beta/mlx';

import {loadTokenizer, loadImageProcessor, loadModel} from './src/index.ts';

const modelDir = 'weights-clip';

main();

async function main() {
  const tokenizer = loadTokenizer(modelDir);
  const imageProcessor = loadImageProcessor(modelDir);
  const model = loadModel(modelDir);

  const pixelValues = await imageProcessor.forward([
    '../mlx-examples/clip/assets/cat.jpeg',
    '../mlx-examples/clip/assets/dog.jpeg',
  ]);
  const output = model.forward({
    inputIds: mx.stack([
      tokenizer.encode('a photo of a cat'),
      tokenizer.encode('a photo of a dog'),
    ]),
    pixelValues,
    returnLoss: true,
  });
  console.log('Loss:', output.loss);
}
