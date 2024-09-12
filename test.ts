import {nn} from '@frost-beta/mlx';

import Clip from './src/index.ts';

main();

async function main() {
  const images = await Promise.all([
    download('https://d29fhpw069ctt2.cloudfront.net/photo/34910/preview/u3x7cekkS16ajjtJcb5L_DSC_5869_npreviews_9e55.jpg'),
    download('https://d29fhpw069ctt2.cloudfront.net/photo/35183/preview/UzWklzFdRBSbkRKhEnvc_1-6128_npreviews_79e3.jpg'),
  ]);

  const clip = new Clip(process.argv[2] ?? 'weights-clip');
  const output = await clip.computeEmbeddings({
    labels: [
      'a photo of a bird',
      'a photo of a dog',
    ],
    images,
  });

  console.log('Cosine similarity:',
              nn.losses.cosineSimilarityLoss(output.labelEmbeddings,
                                             output.imageEmbeddings).tolist());
}

async function download(url) {
  const response = await fetch(url);
  return Buffer.from(await response.arrayBuffer());
}
