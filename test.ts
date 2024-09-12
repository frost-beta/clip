import {nn} from '@frost-beta/mlx';

import {loadTokenizer, loadImageProcessor, loadModel} from './src/index.ts';

const modelDir = 'weights-clip';

main();

async function main() {
  const tokenizer = loadTokenizer(modelDir);
  const imageProcessor = loadImageProcessor(modelDir);
  const model = loadModel(modelDir);

  const output = model.forward({
    inputIds: tokenizer.encode([
      'a photo of a cat',
      'a photo of a dog',
    ]),
    pixelValues: await imageProcessor.forward([
      '../mlx-examples/clip/assets/cat.jpeg',
      '../mlx-examples/clip/assets/dog.jpeg',
    ]),
  });

  console.log('Cosine similarity:',
              nn.losses.cosineSimilarityLoss(output.textEmbeds,
                                             output.imageEmbeds));
}
