import sharp from 'sharp';
import {core as mx} from '@frost-beta/mlx';

export interface PreprocessorConfig {
  cropSize: number,
  doCenterCrop: boolean,
  doNormalize: boolean,
  doResize: boolean,
  imageMean: number[],
  imageStd: number[],
  size: number
}

export class ClipImageProcessor {
  constructor(private config: PreprocessorConfig) {}

  async forward(paths: string[]) {
    const tensors = await Promise.all(paths.map(i => this.preprocess(i)));
    return mx.stack(tensors);
  }

  private async preprocess(path: string) {
    let image = sharp(path);
    if (this.config.doResize)
      image = image.resize(this.config.size, this.config.size, {fit: 'outside'});
    if (this.config.doCenterCrop)
      image = await centerCrop(image, this.config.cropSize);
    const {info, data} = await image.raw().toBuffer({resolveWithObject: true});
    let tensor = mx.array(Array.from(data));
    tensor = tensor.reshape([ info.width, info.height, 3 ]);
    tensor = rescale(tensor);
    if (this.config.doNormalize)
      tensor = normalize(tensor, this.config.imageMean, this.config.imageStd);
    return tensor;
  }
}

async function centerCrop(image: sharp.Sharp, cropSize: number) {
  const {info} = await image.toBuffer({resolveWithObject: true});
  return image.extract({
    top: (info.height - cropSize) / 2,
    left: (info.width - cropSize) / 2,
    width: cropSize,
    height: cropSize,
  });
}

function rescale(tensor: mx.array) {
  return mx.divide(tensor.astype(mx.float32),  255);
}

function normalize(tensor: mx.array, mean: number[], std: number[]) {
  return mx.divide(mx.subtract(tensor, mx.array(mean)), mx.array(std));
}
