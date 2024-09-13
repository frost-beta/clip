import {readFileSync} from 'node:fs'
import {TokenizerLoader} from '@lenml/tokenizers';
import {core as mx, nn} from '@frost-beta/mlx';

import {ClipConfig, ClipModelInput, ClipModel} from './model';
import {PreprocessorConfig, ClipImageProcessor} from './image-processor';

export * from './model';

export interface ClipInput {
  labels?: string[];
  images?: Buffer[];
}

export interface ClipOutput {
  labelEmbeddings?: mx.array;
  imageEmbeddings?: mx.array;
}

/**
 * Provide APIs around the CLIP model.
 */
export class Clip {
  tokenizer: Tokenizer;
  imageProcessor: ClipImageProcessor;
  model: ClipModel;

  constructor(modelDir: string) {
    this.tokenizer = loadTokenizer(modelDir);
    this.imageProcessor = loadImageProcessor(modelDir);
    this.model = loadModel(modelDir);
  }

  async computeEmbeddings({labels, images}: ClipInput): Promise<ClipOutput> {
    const input: ClipModelInput = {};
    if (labels)
      input.inputIds = this.tokenizer.encode(labels);
    if (images)
      input.pixelValues = await this.imageProcessor.forward(images);
    const output = this.model.forward(input);
    return {
      labelEmbeddings: output.textEmbeds,
      imageEmbeddings: output.imageEmbeds,
    };
  }

  static computeCosineSimilaritiy(a1: mx.array, a2: mx.array): number {
    return nn.losses.cosineSimilarityLoss(a1, a2, 0).item() as number;
  }

  static computeCosineSimilarities(x1: mx.array, x2: mx.array): number[] {
    return nn.losses.cosineSimilarityLoss(x1, x2, 1).tolist() as number[];
  }
}

// The tokenizer for encoding multiple strings.
export interface Tokenizer {
  encode(text: string[]): mx.array;
}

// Return the tokenizer.
export function loadTokenizer(dir: string): Tokenizer {
  const tokenizer = TokenizerLoader.fromPreTrained({
    tokenizerJSON: readJson(`${dir}/tokenizer.json`),
    tokenizerConfig: readJson(`${dir}/tokenizer_config.json`),
  });
  return {
    encode(text: string[]) {
      const {input_ids} = tokenizer._call(text, {padding: true});
      return mx.stack(input_ids as number[][]);
    }
  };
}

// Return the image processor.
export function loadImageProcessor(dir: string) {
  const json = readJson(`${dir}/preprocessor_config.json`);
  return new ClipImageProcessor(modelArgs(json) as PreprocessorConfig);
}

// Create the CLIP model.
export function loadModel(dir: string) {
  // Read config files.
  const configJson = readJson(`${dir}/config.json`);
  const clipConfig = modelArgs(configJson) as ClipConfig;
  // Create model.
  const model = new ClipModel(clipConfig);
  const weights = Object.entries(mx.load(`${dir}/model.safetensors`));
  // Sanitize the weights for MLX.
  const sanitizedWeights = [];
  for (const [ key, value ] of weights) {
    // Remove unused position_ids.
    if (key.includes('position_ids'))
      continue;
    // PyTorch Conv2d expects the weight tensor to be of shape:
    // [out_channels, in_channels, kH, KW]
    // MLX Conv2d expects the weight tensor to be of shape:
    // [out_channels, kH, KW, in_channels]
    if (key.endsWith('patch_embedding.weight'))
      sanitizedWeights.push([ key, value.transpose(0, 2, 3, 1) ]);
    else
      sanitizedWeights.push([ key, value ]);
  }
  model.loadWeights(sanitizedWeights);
  return model;
}

// Convert snake_case args into camelCase args.
function modelArgs(args: any): object{
  if (Array.isArray(args))
    return args.map(v => modelArgs(v));
  if (typeof args != 'object')
    return args;
  const newArgs = {}
  for (const key in args) {
    const newKey = key.replace(/(\_\w)/g, (s) => s[1].toUpperCase())
    newArgs[newKey] = modelArgs(args[key]);
  }
  return newArgs
}

// Helper for reading a .json file.
function readJson(path: string) {
  return JSON.parse(String(readFileSync(path)));
}
