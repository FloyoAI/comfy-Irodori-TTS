from . import (
    emoji_picker,
    irodori_cfg_config,
    irodori_lora_stack,
    irodori_model_loader,
    irodori_reference_audio,
    irodori_rescale_config,
    irodori_sampler,
    irodori_voice_design_config,
)

nodes = [
    emoji_picker.EmojiPicker,
    irodori_model_loader.IrodoriModelLoader,
    irodori_lora_stack.IrodoriLoRAStack,
    irodori_reference_audio.IrodoriReferenceAudio,
    irodori_voice_design_config.IrodoriVoiceDesignConfig,
    irodori_cfg_config.IrodoriCFGConfig,
    irodori_rescale_config.IrodoriRescaleConfig,
    irodori_sampler.IrodoriTTSSampler,
]
