import {loadTokenizer, loadModel} from './src/index.ts';

const tokenizer = loadTokenizer('weights-clip');
console.log(loadModel('weights-clip'));
