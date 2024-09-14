import sharp from 'sharp';
import {core as mx} from '@frost-beta/mlx';

export type ImageInputType = Buffer | ArrayBuffer | string;

export interface PreprocessorConfig {
  cropSize: number,
  doCenterCrop: boolean,
  doNormalize: boolean,
  doResize: boolean,
  imageMean: number[],
  imageStd: number[],
  size: number
}

export interface ProcessedImage {
  data: Buffer;
  info: sharp.OutputInfo;
}

export class ClipImageProcessor {
  constructor(private config: PreprocessorConfig) {}

  async processImage(input: ImageInputType): Promise<ProcessedImage> {
    let image = sharp(input);
    if (this.config.doResize && this.config.doCenterCrop && this.config.size == this.config.cropSize) {
      // Fast path for resize and crop with same size.
      image = image.resize(this.config.size, this.config.size);
    } else {
      // Slow path for doing resize and crop in 2 separate steps.
      if (this.config.doResize)
        image = image.resize(this.config.size, this.config.size, {fit: 'outside'});
      if (this.config.doCenterCrop)
        image = await centerCrop(image, this.config.cropSize);
    }
    // The model only works with RGB.
    image = image.removeAlpha();
    // Extract size and data.
    return await image.raw().toBuffer({resolveWithObject: true});
  }

  processImages(inputs: ImageInputType[]): Promise<ProcessedImage[]> {
    return Promise.all(inputs.map(this.processImage.bind(this)));
  }

  normalizeImages(images: ProcessedImage[]) {
    const {info} = images[0];
    // The model expects the data to be a nested array.
    let tensor = mx.stack(images.map(i => mx.array(Array.from(i.data))));
    tensor = tensor.reshape([ images.length, info.width, info.height, 3 ]);
    // Normalize the tensor.
    tensor = rescale(tensor);
    if (this.config.doNormalize)
      tensor = normalize(tensor, this.config.imageMean, this.config.imageStd);
    return tensor;
  }
}

async function centerCrop(image: sharp.Sharp, cropSize: number) {
  // Have to call toBuffer to get the new size after resize.
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
  return mx.divide(mx.subtract(tensor, mx.array(mean)),
                   mx.array(std));
}
