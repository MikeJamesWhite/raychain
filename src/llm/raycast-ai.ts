import { BaseLLM, BaseLLMParams } from 'langchain/llms/base';
import { LLMResult } from 'langchain/schema';
import { AI } from '@raycast/api';
import { RaycastAICallOptions, RaycastAIInput } from '../types';

/**
 * LangChain LLM wrapper around the Raycast AI API.
 */
export class RaycastAI extends BaseLLM<RaycastAICallOptions> implements RaycastAIInput {
  lc_serializable = true;
  model: AI.Model = 'text-davinci-003';
  creativity: AI.Creativity = 'medium';

  get callKeys(): (keyof RaycastAICallOptions)[] {
    return [...(super.callKeys as (keyof RaycastAICallOptions)[])];
  }

  constructor(fields?: Partial<RaycastAIInput> & BaseLLMParams) {
    super(fields ?? {});

    this.model = fields?.model ?? this.model;
    this.creativity = fields?.creativity ?? this.creativity;
  }

  /**
   * Invoke the Raycast AI API with k unique prompts
   *
   * @param [prompts] - The prompts to pass into the model.
   *
   * @returns The full LLM output.
   *
   * @example
   * ```ts
   * import { RaycastAI } from "raychain";
   * const raycastAI = new RaycastAI();
   * const response = await raycastAI.generate(["Tell me a joke."]);
   * ```
   */
  async _generate(prompts: string[]): Promise<LLMResult> {
    const params = this.invocationParams();
    const choices: string[] = [];

    for (let i = 0; i < prompts.length; i += 1) {
      const result = await AI.ask(prompts[i], params);

      choices.push(result);
    }

    const generations = this.chunkArray(choices, 1).map((promptChoices) =>
      promptChoices.map((choice) => ({
        text: choice ?? '',
      }))
    );

    return {
      generations,
    };
  }

  /**
   * Utility method for splitting an array into chunks of a specified size
   */
  private chunkArray<T>(arr: T[], chunkSize: number) {
    return arr.reduce((chunks, elem, index) => {
      const chunkIndex = Math.floor(index / chunkSize);
      const chunk = chunks[chunkIndex] || [];
      chunks[chunkIndex] = chunk.concat([elem]);
      return chunks;
    }, [] as T[][]);
  }

  /**
   * Get the parameters used to invoke the model
   */
  invocationParams() {
    return {
      model: this.model,
      creativity: this.creativity,
    };
  }

  /**
   * Get the identifying parameters for the model
   */
  identifyingParams() {
    return {
      model_name: this.model,
      ...this.invocationParams(),
    };
  }

  _llmType() {
    return 'raycastai';
  }
}
