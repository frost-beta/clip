import sharp from 'sharp';
import {core as mx} from '@frost-beta/mlx';

export type BufferType = Buffer | ArrayBuffer | Uint8Array | Uint8ClampedArray |
                         Int8Array | Uint16Array | Int16Array | Uint32Array |
                         Int32Array | Float32Array | Float64Array;

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

  async forward(buffers: BufferType[]) {
    const tensors = await Promise.all(buffers.map(i => this.preprocess(i)));
    return mx.stack(tensors);
  }

  private async preprocess(buffer: BufferType) {
    let image = sharp(buffer);
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
    const {info, data} = await image.raw().toBuffer({resolveWithObject: true});
    // The model expects the data to be a nested array.
    let tensor = mx.array(Array.from(data));
    tensor = tensor.reshape([ info.width, info.height, 3 ]);
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
  return mx.divide(mx.subtract(tensor, mx.array(mean)), mx.array(std));
}
