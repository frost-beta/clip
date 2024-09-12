import {readFileSync} from 'node:fs'
import {TokenizerLoader} from '@lenml/tokenizers';
import {core as mx} from '@frost-beta/mlx';

import {ClipConfig, ClipModel} from './model';

export * from './model';

// Return a tokenizer.
export function loadTokenizer(dir: string) {
  const tokenizerJSON = readFileSync(`${dir}/tokenizer.json`);
  const tokenizerConfig = readFileSync(`${dir}/tokenizer_config.json`);
  return TokenizerLoader.fromPreTrained({
    tokenizerJSON: JSON.parse(String(tokenizerJSON)),
    tokenizerConfig: JSON.parse(String(tokenizerConfig)),
  });
}

// Create the CLIP model.
export function loadModel(dir: string) {
  // Read config files.
  const configJson = JSON.parse(String(readFileSync(`${dir}/config.json`)));
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
export function modelArgs(args: any): object{
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
