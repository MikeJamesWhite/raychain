import { AI } from '@raycast/api';
import { BaseLLMCallOptions } from 'langchain/llms/base';

export declare interface RaycastAIInput {
  creativity: AI.Creativity;
  model: AI.Model;
}

export type RaycastAICallOptions = BaseLLMCallOptions;
