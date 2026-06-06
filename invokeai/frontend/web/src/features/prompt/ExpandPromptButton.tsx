import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Button,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  spinAnimation,
  Switch,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import {
  ideogram4MagicPromptEnabledChanged,
  ideogram4MagicPromptModelSelected,
  positivePromptChanged,
  selectIdeogram4MagicPromptEnabled,
  selectIdeogram4MagicPromptModel,
  selectMainModelConfig,
  selectPositivePrompt,
} from 'features/controlLayers/store/paramsSlice';
import { setInstallModelsTabByName } from 'features/modelManagerV2/store/installModelsStore';
import { ModelPicker } from 'features/parameters/components/ModelPicker';
import { getPromptExpansionArgs } from 'features/prompt/getPromptExpansionArgs';
import { setPromptUndo } from 'features/prompt/promptUndo';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSparkleBold } from 'react-icons/pi';
import { useExpandPromptMutation } from 'services/api/endpoints/utilities';
import { useTextLLMModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ExpandPromptButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const prompt = useAppSelector(selectPositivePrompt);
  const mainModelConfig = useAppSelector(selectMainModelConfig);
  const magicPromptModel = useAppSelector(selectIdeogram4MagicPromptModel);
  const autoExpandEnabled = useAppSelector(selectIdeogram4MagicPromptEnabled);
  const [modelConfigs] = useTextLLMModels();
  const popover = useDisclosure(false);
  const [localSelectedModel, setLocalSelectedModel] = useState<AnyModelConfig | undefined>(undefined);
  const [expandPrompt, { isLoading }] = useExpandPromptMutation();

  const hasModels = modelConfigs.length > 0;
  // For Ideogram 4 the chosen LLM is persisted (shared with auto-expand on generate); other models keep
  // an ephemeral, click-only selection.
  const isIdeogram4 = mainModelConfig?.base === 'ideogram4';
  const selectedModel = useMemo(
    () => (isIdeogram4 ? modelConfigs.find((config) => config.key === magicPromptModel?.key) : localSelectedModel),
    [isIdeogram4, modelConfigs, magicPromptModel?.key, localSelectedModel]
  );

  const handleModelChange = useCallback(
    (model: AnyModelConfig) => {
      if (isIdeogram4) {
        dispatch(ideogram4MagicPromptModelSelected(model));
      } else {
        setLocalSelectedModel(model);
      }
    },
    [isIdeogram4, dispatch]
  );

  const handleAutoExpandChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(ideogram4MagicPromptEnabledChanged(e.target.checked));
    },
    [dispatch]
  );

  const handleExpand = useCallback(async () => {
    if (!selectedModel || !prompt.trim()) {
      return;
    }
    try {
      const result = await expandPrompt({
        prompt,
        model_key: selectedModel.key,
        ...getPromptExpansionArgs(mainModelConfig?.base),
      }).unwrap();
      if (result.expanded_prompt) {
        setPromptUndo(prompt);
        dispatch(positivePromptChanged(result.expanded_prompt));
      }
      popover.close();
    } catch {
      // Error is handled by RTK Query
    }
  }, [selectedModel, prompt, expandPrompt, mainModelConfig?.base, dispatch, popover]);

  const handleOpenModelManager = useCallback(() => {
    popover.close();
    navigationApi.switchToTab('models');
    setInstallModelsTabByName('starterModels');
  }, [popover]);

  return (
    <Popover
      isOpen={popover.isOpen}
      onOpen={popover.open}
      onClose={popover.close}
      placement="left-start"
      isLazy
      closeOnBlur={false}
    >
      <PopoverTrigger>
        <span>
          <Tooltip label={hasModels ? t('prompt.expandPromptWithLLM') : t('prompt.noTextLLMInstalledTitle')}>
            <IconButton
              size="sm"
              variant="promptOverlay"
              aria-label={t('prompt.expandPromptWithLLM')}
              icon={<PiSparkleBold />}
              sx={isLoading ? loadingStyles : undefined}
              isDisabled={isLoading || (hasModels && !prompt.trim())}
            />
          </Tooltip>
        </span>
      </PopoverTrigger>
      <Portal>
        <PopoverContent p={3} w={350}>
          <PopoverArrow />
          <PopoverBody p={0}>
            {hasModels ? (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.expandPrompt')}
                </Text>
                <ModelPicker
                  pickerId="expand-prompt-model"
                  modelConfigs={modelConfigs}
                  selectedModelConfig={selectedModel}
                  onChange={handleModelChange}
                  placeholder={t('prompt.selectTextLLM')}
                />
                {isIdeogram4 && (
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <Flex w="full" justifyContent="space-between" alignItems="center">
                      <FormLabel m={0}>{t('prompt.autoExpandOnGenerate')}</FormLabel>
                      <Switch isChecked={autoExpandEnabled} onChange={handleAutoExpandChange} />
                    </Flex>
                    <Text fontSize="xs" color="base.300">
                      {t('prompt.autoExpandOnGenerateDescription')}
                    </Text>
                  </FormControl>
                )}
                <Button
                  size="sm"
                  colorScheme="invokeBlue"
                  onClick={handleExpand}
                  isLoading={isLoading}
                  isDisabled={!selectedModel || !prompt.trim()}
                >
                  {t('prompt.expand')}
                </Button>
              </Flex>
            ) : (
              <Flex flexDir="column" gap={3}>
                <Text fontWeight="semibold" fontSize="sm">
                  {t('prompt.noTextLLMInstalledTitle')}
                </Text>
                <Text fontSize="sm" color="base.300">
                  {t('prompt.noTextLLMInstalledDescription')}
                </Text>
                <Button size="sm" colorScheme="invokeBlue" onClick={handleOpenModelManager}>
                  {t('prompt.openModelManager')}
                </Button>
              </Flex>
            )}
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ExpandPromptButton.displayName = 'ExpandPromptButton';
