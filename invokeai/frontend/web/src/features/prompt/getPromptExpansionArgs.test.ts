import { getPromptExpansionArgs } from 'features/prompt/getPromptExpansionArgs';
import { describe, expect, it } from 'vitest';

describe('getPromptExpansionArgs', () => {
  it('returns the Ideogram 4 magic-prompt preset and a larger token budget for ideogram4 models', () => {
    expect(getPromptExpansionArgs('ideogram4')).toEqual({
      system_prompt_preset: 'ideogram4_v1',
      max_tokens: 2048,
    });
  });

  it('returns no extra args for other model bases', () => {
    expect(getPromptExpansionArgs('flux')).toEqual({});
    expect(getPromptExpansionArgs('sdxl')).toEqual({});
  });

  it('returns no extra args when no model is selected', () => {
    expect(getPromptExpansionArgs(undefined)).toEqual({});
  });
});
