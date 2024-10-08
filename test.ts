import {Clip} from './src/index.ts';

main();

async function main() {
  const clip = new Clip(process.argv[2] ?? 'clip-vit-large-patch14');
  const labelEmbeddings = clip.computeLabelEmbeddingsJs([ 'seagull', 'lovely dog' ]);
  const imageEmbeddings = clip.computeImageEmbeddingsJs(
    await clip.processImages(await Promise.all([
      download('https://d29fhpw069ctt2.cloudfront.net/photo/34910/preview/u3x7cekkS16ajjtJcb5L_DSC_5869_npreviews_9e55.jpg'),
      download('https://d29fhpw069ctt2.cloudfront.net/photo/35183/preview/UzWklzFdRBSbkRKhEnvc_1-6128_npreviews_79e3.jpg'),
    ])));
  const [ scores, indices ] = Clip.computeCosineSimilarities(labelEmbeddings,
                                                             imageEmbeddings);
  console.log('Cosine similarity:', scores.tolist());
}

async function download(url) {
  const response = await fetch(url);
  return Buffer.from(await response.arrayBuffer());
}
