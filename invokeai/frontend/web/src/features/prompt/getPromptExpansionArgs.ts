import type { BaseModelType } from 'features/nodes/types/common';

/**
 * Ideogram 4 is trained on structured JSON captions, so a plain prompt must be expanded with the
 * model's magic-prompt system prompt. The caption is long, so it also needs a larger token budget
 * than the generic prompt-expansion default.
 */
const IDEOGRAM4_V1_PRESET = 'ideogram4_v1';
const IDEOGRAM4_MAX_TOKENS = 2048;

type PromptExpansionArgs = {
  system_prompt_preset?: string;
  max_tokens?: number;
};

/**
 * Extra expand-prompt arguments for the selected main model. Models without a magic-prompt preset
 * return an empty object and use the endpoint defaults.
 */
export const getPromptExpansionArgs = (mainModelBase: BaseModelType | undefined): PromptExpansionArgs => {
  if (mainModelBase === 'ideogram4') {
    return { system_prompt_preset: IDEOGRAM4_V1_PRESET, max_tokens: IDEOGRAM4_MAX_TOKENS };
  }
  return {};
};
