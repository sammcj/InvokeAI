import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('app/logging/logger', () => ({
  logger: () => ({
    debug: vi.fn(),
  }),
}));

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const model = {
  key: 'ideogram4-model',
  hash: 'ideogram4-hash',
  name: 'Ideogram 4 (Diffusers)',
  base: 'ideogram4',
  type: 'main',
};

const defaultParams: {
  cfgScale: number | number[];
  steps: number;
} = {
  cfgScale: 7,
  steps: 20,
};

let params = { ...defaultParams };

vi.mock('features/controlLayers/store/paramsSlice', () => ({
  selectMainModelConfig: vi.fn(() => model),
  selectParamsSlice: vi.fn(() => params),
}));

vi.mock('features/controlLayers/store/selectors', () => ({
  selectCanvasMetadata: vi.fn(() => ({})),
}));

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(model)),
}));

vi.mock('features/nodes/util/graph/generation/addImageToImage', () => ({
  addImageToImage: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addInpaint', () => ({
  addInpaint: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addNSFWChecker', () => ({
  addNSFWChecker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/generation/addOutpaint', () => ({
  addOutpaint: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addTextToImage', () => ({
  addTextToImage: vi.fn(({ l2i }) => l2i),
}));

vi.mock('features/nodes/util/graph/generation/addWatermarker', () => ({
  addWatermarker: vi.fn((_g, node) => node),
}));

vi.mock('features/nodes/util/graph/graphBuilderUtils', () => ({
  selectCanvasOutputFields: vi.fn(() => ({})),
}));

vi.mock('features/ui/store/uiSelectors', () => ({
  selectActiveTab: vi.fn(() => 'generation'),
}));

vi.mock('services/api/types', async () => {
  const actual = await vi.importActual('services/api/types');
  return {
    ...actual,
    isNonRefinerMainModelConfig: vi.fn(() => true),
  };
});

import { buildIdeogram4Graph } from './buildIdeogram4Graph';

const buildState = () =>
  ({
    system: {
      shouldUseNSFWChecker: false,
      shouldUseWatermarker: false,
    },
  }) as never;

describe('buildIdeogram4Graph', () => {
  afterEach(() => {
    nextId = 0;
    params = { ...defaultParams };
  });

  it('includes the model loader, text encoder, denoise and vae decode nodes', async () => {
    const { g } = await buildIdeogram4Graph({ generationMode: 'txt2img', manager: null, state: buildState() });

    const nodeTypes = Object.values(g.getGraph().nodes).map((n) => n.type);
    expect(nodeTypes).toContain('ideogram4_model_loader');
    expect(nodeTypes).toContain('ideogram4_text_encoder');
    expect(nodeTypes).toContain('ideogram4_denoise');
    expect(nodeTypes).toContain('ideogram4_vae_decode');
  });

  it('wires both the conditional and unconditional transformers to the denoise node', async () => {
    const { g } = await buildIdeogram4Graph({ generationMode: 'txt2img', manager: null, state: buildState() });

    const edges = g.getGraph().edges;
    expect(edges.some((e) => e.destination.field === 'transformer')).toBe(true);
    expect(edges.some((e) => e.destination.field === 'unconditional_transformer')).toBe(true);
    // Asymmetric CFG means a single positive conditioning, no negative conditioning.
    expect(edges.some((e) => e.destination.field === 'conditioning')).toBe(true);
    expect(edges.some((e) => e.destination.field === 'negative_conditioning')).toBe(false);
  });

  it('passes steps and guidance through to the denoise node', async () => {
    params = { ...defaultParams, steps: 24, cfgScale: 6 };
    const { g } = await buildIdeogram4Graph({ generationMode: 'txt2img', manager: null, state: buildState() });

    const denoise = Object.values(g.getGraph().nodes).find((n) => n.type === 'ideogram4_denoise');
    expect(denoise).toMatchObject({ num_steps: 24, guidance_scale: 6 });
  });

  it('adds the image-to-latents node for img2img', async () => {
    const { g } = await buildIdeogram4Graph({
      generationMode: 'img2img',
      manager: { id: 'test-manager' } as never,
      state: buildState(),
    });

    const nodeTypes = Object.values(g.getGraph().nodes).map((n) => n.type);
    expect(nodeTypes).toContain('ideogram4_i2l');
  });
});
