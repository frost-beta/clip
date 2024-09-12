import {core as mx, nn} from '@frost-beta/mlx';

export interface ClipConfig {
  textConfig: ClipTextConfig;
  visionConfig: ClipVisionConfig;
  projectionDim: number;
}

interface EncoderConfig {
  numHiddenLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numAttentionHeads: number;
  layerNormEps: number;
}

export interface ClipTextConfig extends EncoderConfig {
  maxPositionEmbeddings: number;
  vocabSize: number;
}

export interface ClipVisionConfig extends EncoderConfig {
  numChannels: number;
  imageSize: number;
  patchSize: number;
}

export interface ClipTextOutput {
  poolerOutput: mx.array;
  lastHiddenState: mx.array;
}

export interface ClipVisionOutput extends ClipTextOutput {
  hiddenStates?: mx.array;
}

export interface ClipModelOutput {
  loss?: mx.array;
  textEmbeds?: mx.array;
  imageEmbeds?: mx.array;
  textModelOutput?: ClipTextOutput;
  visionModelOutput?: ClipVisionOutput;
}

class Attention extends nn.Module {
  numHeads: number;
  qProj: nn.Linear;
  kProj: nn.Linear;
  vProj: nn.Linear;
  outProj: nn.Linear;

  constructor(dims: number,
              numHeads: number,
              queryInputDims: number | null = null,
              keyInputDims: number | null = null,
              valueInputDims: number | null = null,
              valueDims: number | null = null,
              valueOutputDims: number | null = null,
              bias: boolean = true) {
    if (dims % numHeads != 0) {
      throw new Error(`The input feature dimensions should be divisible by the ` +
                      `number of heads (${dims} % ${numHeads}) != 0`);
    }

    super();

    queryInputDims = queryInputDims || dims;
    keyInputDims = keyInputDims || dims;
    valueInputDims = valueInputDims || keyInputDims;
    valueDims = valueDims || dims;
    valueOutputDims = valueOutputDims || dims;

    this.numHeads = numHeads;
    this.qProj = new nn.Linear(queryInputDims, dims, bias);
    this.kProj = new nn.Linear(keyInputDims, dims, bias);
    this.vProj = new nn.Linear(valueInputDims, valueDims, bias);
    this.outProj = new nn.Linear(valueDims, valueOutputDims, bias);
  }

  forward(queries: mx.array, keys: mx.array, values: mx.array, mask?: mx.array) {
    queries = this.qProj.forward(queries);
    keys = this.kProj.forward(keys);
    values = this.vProj.forward(values);

    const numHeads = this.numHeads;
    const [ B, L, D ] = queries.shape;
    const [  , S,   ] = keys.shape;
    queries = queries.reshape(B, L, numHeads, -1).transpose(0, 2, 1, 3);
    keys = keys.reshape(B, S, numHeads, -1).transpose(0, 2, 3, 1);
    values = values.reshape(B, S, numHeads, -1).transpose(0, 2, 1, 3);

    const scale = Math.sqrt(1 / queries.shape.at(-1));
    let scores = mx.matmul(mx.multiply(queries, scale), keys);
    if (mask)
      scores = mx.add(scores, mask.astype(scores.dtype));
    scores = mx.softmax(scores, -1);
    const valuesHat = mx.matmul(scores, values).transpose(0, 2, 1, 3)
                                               .reshape(B, L, -1);

    return this.outProj.forward(valuesHat);
  }
}

class MLP extends nn.Module {
  activationFn: (x: mx.array) => mx.array;
  fc1: nn.Linear;
  fc2: nn.Linear;

  constructor(config: EncoderConfig) {
    super();
    this.activationFn = quickGelu;
    this.fc1 = new nn.Linear(config.hiddenSize, config.intermediateSize);
    this.fc2 = new nn.Linear(config.intermediateSize, config.hiddenSize);
  }

  forward(x: mx.array): mx.array {
    x = this.activationFn(this.fc1.forward(x));
    x = this.fc2.forward(x);
    return x;
  }
}

class EncoderLayer extends nn.Module {
  embedDim: number;
  selfAttn: Attention;
  layerNorm1: nn.LayerNorm;
  mlp: MLP;
  layerNorm2: nn.LayerNorm;

  constructor(config: EncoderConfig) {
    super();
    this.embedDim = config.hiddenSize;
    this.selfAttn = new Attention(config.hiddenSize, config.numAttentionHeads);
    this.layerNorm1 = new nn.LayerNorm(this.embedDim, config.layerNormEps);
    this.mlp = new MLP(config);
    this.layerNorm2 = new nn.LayerNorm(this.embedDim, config.layerNormEps);
  }

  forward(x: mx.array, mask?: mx.array): mx.array {
    let y = this.layerNorm1.forward(x);
    y = this.selfAttn.forward(y, y, y, mask);
    x = mx.add(x, y);
    y = this.layerNorm2.forward(x);
    y = this.mlp.forward(y);
    return mx.add(x, y);
  }
}

class Encoder extends nn.Module {
  layers: EncoderLayer[] = [];

  constructor(config: EncoderConfig) {
    super();
    for (let i = 0; i < config.numHiddenLayers; ++i)
      this.layers.push(new EncoderLayer(config))
  }

  forward(h: mx.array, mask?: mx.array) {
    for (const layer of this.layers)
      h = layer.forward(h, mask);
    return h;
  }
}

class TextEmbeddings extends nn.Module {
  tokenEmbedding: nn.Embedding;
  positionEmbedding: nn.Embedding;

  constructor(config: ClipTextConfig) {
    super();
    const embedDim = config.hiddenSize;
    this.tokenEmbedding = new nn.Embedding(config.vocabSize, embedDim);
    this.positionEmbedding = new nn.Embedding(config.maxPositionEmbeddings, embedDim);
  }

  forward(x: mx.array): mx.array {
    const embeddings = this.tokenEmbedding.forward(x.astype(mx.int32));
    return mx.add(embeddings,
                  this.positionEmbedding.weight.index(mx.Slice(null, x.shape[1])));
  }
}

/**
 * Implements the text encoder transformer from CLIP.
 */
export class ClipTextModel extends nn.Module {
  embeddings: TextEmbeddings;
  encoder: Encoder;
  finalLayerNorm: nn.LayerNorm;

  constructor(config: ClipTextConfig) {
    super();
    this.embeddings = new TextEmbeddings(config);
    this.encoder = new Encoder(config);
    this.finalLayerNorm = new nn.LayerNorm(config.hiddenSize);
  }

  forward(x: mx.array): ClipTextOutput {
    const [ B, N ] = x.shape;
    const eotTokens = mx.argmax(x, -1);
    x = this.embeddings.forward(x);
    const mask = nn.MultiHeadAttention.createAdditiveCausalMask(N, x.dtype);
    x = this.encoder.forward(x, mask);
    const lastHiddenState = this.finalLayerNorm.forward(x);
    const poolerOutput = lastHiddenState.index(mx.arange(B, mx.int32), eotTokens);
    return {poolerOutput, lastHiddenState};
  }
}

class VisionEmbeddings extends nn.Module {
  embedDim: number;
  imageSize: number;
  patchSize: number;
  classEmbedding: mx.array;
  patchEmbedding: nn.Conv2d;
  numPatches: number;
  numPositions: number;
  positionEmbedding: nn.Embedding;

  constructor(config: ClipVisionConfig) {
    super();
    this.embedDim = config.hiddenSize;
    this.imageSize = config.imageSize;
    this.patchSize = config.patchSize;

    this.classEmbedding = mx.zeros(config.hiddenSize);

    this.patchEmbedding = new nn.Conv2d(3,
                                        this.embedDim,
                                        this.patchSize,
                                        this.patchSize,
                                        undefined,
                                        undefined,
                                        false);

    this.numPatches = Math.pow(this.imageSize / this.patchSize, 2);
    this.numPositions = this.numPatches + 1;
    this.positionEmbedding = new nn.Embedding(this.numPositions, this.embedDim);
  }

  forward(x: mx.array): mx.array {
    const batchSize = x.shape[0];
    // Patchify using conv:
    // [batch_size, sqrt(num_patches), sqrt(num_patches), embed_dim]
    let patchEmbeddings = this.patchEmbedding.forward(x);
    // [batch_size, num_patches, embed_dim]
    patchEmbeddings = mx.flatten(patchEmbeddings, 1, 2);
    const embedDim = patchEmbeddings.shape.at(-1);
    // Prepend <CLS> embeddings
    // [batch_size, 1, embed_dim]
    const clsEmbeddings = mx.broadcastTo(this.classEmbedding,
                                         [ batchSize, 1, embedDim ]);
    // [batch_size, num_patches + 1, embed_dim]
    let embeddings = mx.concatenate([ clsEmbeddings, patchEmbeddings ], 1);
    // Add positional encoding
    embeddings = mx.add(embeddings, this.positionEmbedding.weight);
    return embeddings;
  }
}

/**
 * Implements the vision encoder transformer from CLIP.
 */
export class ClipVisionModel extends nn.Module {
  embeddings: VisionEmbeddings;
  preLayrnorm: nn.LayerNorm;
  encoder: Encoder;
  postLayernorm: nn.LayerNorm;

  constructor(config: ClipVisionConfig) {
    super();
    this.embeddings = new VisionEmbeddings(config);
    this.preLayrnorm = new nn.LayerNorm(config.hiddenSize);
    this.encoder = new Encoder(config);
    this.postLayernorm = new nn.LayerNorm(config.hiddenSize);
  }

  forward(x: mx.array, outputHiddenStates = false): ClipVisionOutput {
    x = this.embeddings.forward(x);
    x = this.preLayrnorm.forward(x);

    const encoderStates = [ x ];
    for (const layer of this.encoder.layers) {
      x = layer.forward(x);
      encoderStates.push(x);
    }

    const poolerOutput = this.postLayernorm.forward(x.index(mx.Slice(), 1, mx.Slice()));
    return {
      poolerOutput,
      lastHiddenState: x,
      hiddenStates: outputHiddenStates ? mx.stack(encoderStates) : undefined,
    };
  }
}

export class ClipModel extends nn.Module {
  textModel: ClipTextModel;
  visionModel: ClipVisionModel;
  textEmbedDim: number;
  visionEmbedDim: number;
  projectionDim: number;
  visualProjection: nn.Linear;
  textProjection: nn.Linear;
  logitScale: mx.array;

  constructor(config: ClipConfig) {
    super();
    this.textModel = new ClipTextModel(config.textConfig);
    this.visionModel = new ClipVisionModel(config.visionConfig);

    this.textEmbedDim = config.textConfig.hiddenSize;
    this.visionEmbedDim = config.visionConfig.hiddenSize;
    this.projectionDim = config.projectionDim;

    this.visualProjection = new nn.Linear(this.visionEmbedDim, this.projectionDim, false);
    this.textProjection = new nn.Linear(this.textEmbedDim, this.projectionDim, false);
    this.logitScale = mx.array(0.);
  }

  getTextFeatures(x: mx.array): mx.array {
    return this.textProjection.forward(this.textModel.forward(x).poolerOutput);
  }

  getImageFeatures(x: mx.array): mx.array {
    return this.visualProjection.forward(this.visionModel.forward(x).poolerOutput);
  }

  forward({inputIds, pixelValues, returnLoss} : {inputIds?: mx.array, pixelValues?: mx.array, returnLoss?: boolean}): ClipModelOutput {
    let textEmbeds, textModelOutput, imageEmbeds, visionModelOutput;
    if (inputIds) {
      textModelOutput = this.textModel.forward(inputIds);
      textEmbeds = this.textProjection.forward(textModelOutput.poolerOutput);
      textEmbeds = mx.divide(textEmbeds, mx.linalg.norm(textEmbeds, undefined, -1, true));
    }
    if (pixelValues) {
      visionModelOutput = this.visionModel.forward(pixelValues);
      imageEmbeds = this.visualProjection.forward(visionModelOutput.poolerOutput);
      imageEmbeds = mx.divide(imageEmbeds, mx.linalg.norm(imageEmbeds, undefined, -1, true));
    }

    if (returnLoss && (!inputIds || !pixelValues)) {
      throw new Error("Must provide text and image inputs to compute loss.");
    }

    let loss;
    if (returnLoss) {
      const logits = mx.multiply(mx.matmul(textEmbeds, imageEmbeds.T),
                                 mx.exp(this.logitScale));
      loss = clipLoss(logits);
    }

    return {loss, textEmbeds, imageEmbeds, visionModelOutput, textModelOutput};
  }
}

// A fast GELU approximation https://github.com/hendrycks/GELUs
function quickGelu(x: mx.array): mx.array {
  return mx.multiply(x, mx.sigmoid(mx.multiply(1.702, x)));
}

// Compute loss of CLIP model's output.
function clipLoss(logits: mx.array): mx.array {
  const [ N, M ] = logits.shape;
  const captionLoss = nn.losses.crossEntropy(logits, mx.arange(N, mx.int32), undefined, undefined, undefined, 'mean');
  const imageLoss = nn.losses.crossEntropy(logits.T, mx.arange(M, mx.int32), undefined, undefined, undefined, 'mean');
  return mx.divide(mx.add(captionLoss, imageLoss),
                   2.0);
}
