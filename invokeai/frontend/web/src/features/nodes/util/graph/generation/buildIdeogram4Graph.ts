import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata } from 'features/controlLayers/store/selectors';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { selectTextLLMModels } from 'services/api/hooks/modelsByType';
import type { Invocation } from 'services/api/types';
import { isNonRefinerMainModelConfig } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const log = logger('system');

export const buildIdeogram4Graph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Ideogram 4 graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model selected');
  assert(model.base === 'ideogram4', 'Selected model is not an Ideogram 4 model');

  const params = selectParamsSlice(state);
  const { cfgScale: guidance_scale, steps, ideogram4MagicPromptEnabled, ideogram4MagicPromptModel } = params;

  const g = new Graph(getPrefixedId('ideogram4_graph'));

  // The Ideogram 4 model is a single diffusers pipeline: the loader exposes the conditional and
  // unconditional transformers, the Qwen3-VL encoder and the VAE.
  const modelLoader = g.addNode({
    type: 'ideogram4_model_loader',
    id: getPrefixedId('ideogram4_model_loader'),
    model,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'ideogram4_text_encoder',
    id: getPrefixedId('pos_prompt'),
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  // Ideogram 4 uses asymmetric CFG via a separate unconditional transformer, so there is no negative
  // prompt; guidance_scale blends the conditional and unconditional velocity predictions.
  const denoise = g.addNode({
    type: 'ideogram4_denoise',
    id: getPrefixedId('ideogram4_denoise'),
    num_steps: steps,
    guidance_scale,
  });
  const l2i = g.addNode({
    type: 'ideogram4_vae_decode',
    id: getPrefixedId('ideogram4_vae_decode'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 'unconditional_transformer', denoise, 'unconditional_transformer');
  g.addEdge(modelLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  // Ideogram 4 is trained on structured JSON captions, so a plain prompt is expanded by a local text LLM
  // ("magic prompt") before it reaches the encoder. Without expansion the model tends to return its
  // baked-in safety-filter card. Resolve the LLM to use: the explicitly chosen one if still installed,
  // otherwise auto-select the first available text LLM so generation works without manual setup. Only when
  // expansion is disabled or no text LLM is installed at all does the prompt pass through unexpanded.
  const textLLMModels = selectTextLLMModels(state);
  const magicPromptModel =
    textLLMModels.find((m) => m.key === ideogram4MagicPromptModel?.key) ?? textLLMModels.at(0) ?? null;
  if (ideogram4MagicPromptEnabled && magicPromptModel) {
    const expandPrompt = g.addNode({
      type: 'ideogram4_expand_prompt',
      id: getPrefixedId('ideogram4_expand_prompt'),
      text_llm_model: magicPromptModel,
    });
    g.addEdge(positivePrompt, 'value', expandPrompt, 'prompt');
    g.addEdge(expandPrompt, 'value', posCond, 'prompt');
  } else {
    g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  }
  g.addEdge(posCond, 'conditioning', denoise, 'conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isNonRefinerMainModelConfig);
  assert(modelConfig.base === 'ideogram4');

  g.upsertMetadata({
    cfg_scale: guidance_scale,
    model: Graph.getModelMetadataField(modelConfig),
    steps,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({ g, state, denoise, l2i });
    g.upsertMetadata({ generation_mode: 'ideogram4_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({ type: 'ideogram4_i2l', id: getPrefixedId('ideogram4_i2l') });
    canvasOutput = await addImageToImage({
      g,
      state,
      manager,
      denoise,
      l2i,
      i2l,
      vaeSource: modelLoader,
    });
    g.upsertMetadata({ generation_mode: 'ideogram4_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({ type: 'ideogram4_i2l', id: getPrefixedId('ideogram4_i2l') });
    canvasOutput = await addInpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'ideogram4_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({ type: 'ideogram4_i2l', id: getPrefixedId('ideogram4_i2l') });
    canvasOutput = await addOutpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'ideogram4_outpaint' });
  } else {
    assert<Equals<typeof generationMode, never>>(false);
  }

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }

  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));

  if (selectActiveTab(state) === 'canvas') {
    g.upsertMetadata(selectCanvasMetadata(state));
  }

  g.setMetadataReceivingNode(canvasOutput);

  return {
    g,
    seed,
    positivePrompt,
  };
};
