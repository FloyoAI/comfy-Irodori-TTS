import sys

import comfy.utils
import torch
from comfy_api.latest import io

from ...modules.irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    get_cached_runtime,
)
from ...node_utils import mk_name
from .common import CATEGORY, PACKAGE_NAME
from .irodori_common import (
    IO_CFG_CONFIG,
    IO_LORA_STACK,
    IO_MODEL_CONFIG,
    IO_REF_CONFIG,
    IO_RESCALE_CONFIG,
    IO_VOICE_DESIGN_CONFIG,
    none_if_non_positive,
)


class IrodoriTTSSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=mk_name(PACKAGE_NAME, "Sampler"),
            display_name="IrodoriTTS Sampler",
            category=CATEGORY,
            inputs=[
                IO_MODEL_CONFIG.Input(
                    "model_config",
                    display_name="irodori_model_config",
                    tooltip="IrodoriTTS Model Loaderの出力を接続します。",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip="生成する読み上げテキストです。",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=sys.maxsize,
                    tooltip="生成に使用する乱数seedです。同じ設定なら同じ結果を再現します。",
                ),
                io.Float.Input(
                    "seconds",
                    default=0.0,
                    min=0.0,
                    max=120.0,
                    step=0.5,
                    tooltip="生成する音声長です。0ならv3モデルでは自動推定し、duration predictorのないモデルでは30秒にフォールバックします。",
                ),
                io.Float.Input(
                    "duration_scale",
                    default=1.0,
                    min=0.1,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                    tooltip="v3自動秒数推定の倍率です。secondsが0のときに有効で、1より大きいと長く、小さいと短くなります。",
                ),
                io.Float.Input(
                    "min_seconds",
                    default=0.5,
                    min=0.1,
                    max=120.0,
                    step=0.1,
                    advanced=True,
                    tooltip="v3自動秒数推定で許可する最短秒数です。",
                ),
                io.Float.Input(
                    "max_seconds",
                    default=30.0,
                    min=0.1,
                    max=120.0,
                    step=0.5,
                    advanced=True,
                    tooltip="v3自動秒数推定で許可する最長秒数です。secondsを手動指定した場合もこの範囲に丸めます。",
                ),
                io.Int.Input(
                    "num_steps",
                    default=40,
                    min=1,
                    max=120,
                    tooltip="サンプリングステップ数です。大きいほど遅くなります。",
                ),
                IO_LORA_STACK.Input(
                    "lora_stack",
                    optional=True,
                    tooltip="IrodoriTTS LoRA Stackの出力を接続します。未接続ならLoRAなしで生成します。",
                ),
                IO_REF_CONFIG.Input(
                    "ref_config",
                    optional=True,
                    tooltip="話者参照音声の設定です。未接続なら参照なし生成になります。",
                ),
                IO_VOICE_DESIGN_CONFIG.Input(
                    "voice_design_config",
                    optional=True,
                    tooltip="VoiceDesignモデル用のcaption設定です。通常モデルでは接続不要です。",
                ),
                IO_CFG_CONFIG.Input(
                    "cfg_config",
                    optional=True,
                    tooltip="CFGの詳細設定です。未接続なら標準値を使用します。",
                ),
                IO_RESCALE_CONFIG.Input(
                    "rescale_config",
                    optional=True,
                    tooltip="rescaleやspeaker K/V補正の設定です。未接続なら無効です。",
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=16,
                    advanced=True,
                    tooltip="同じ条件で同時生成する音声のバッチ数です。出力AUDIOのbatch方向に複数候補を格納します。",
                ),
                io.Combo.Input(
                    "decode_mode",
                    options=["sequential", "batch"],
                    default="sequential",
                    advanced=True,
                    tooltip="codecデコード方式です。batchは速い場合がありますがVRAMを多く使います。",
                ),
                io.Boolean.Input(
                    "context_kv_cache",
                    default=True,
                    advanced=True,
                    tooltip="コンテキストK/Vキャッシュを使用します。通常は有効を推奨します。",
                ),
                io.Int.Input(
                    "max_text_len",
                    default=0,
                    min=0,
                    max=4096,
                    advanced=True,
                    tooltip="テキストtoken長の上限です。0ならチェックポイント既定値を使います。",
                ),
                io.Boolean.Input(
                    "trim_tail",
                    default=True,
                    advanced=True,
                    tooltip="末尾の無音や平坦化した部分を推定して切り詰めます。",
                ),
                io.Int.Input(
                    "tail_window_size",
                    default=20,
                    min=1,
                    max=200,
                    advanced=True,
                    tooltip="末尾切り詰め判定に使う潜在窓サイズです。",
                ),
                io.Float.Input(
                    "tail_std_threshold",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="末尾切り詰め判定の標準偏差しきい値です。",
                ),
                io.Float.Input(
                    "tail_mean_threshold",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="末尾切り詰め判定の平均値しきい値です。",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_config: dict,
        text: str,
        seed: int,
        seconds: float,
        duration_scale: float,
        min_seconds: float,
        max_seconds: float,
        num_steps: int,
        batch_size: int,
        decode_mode: str,
        context_kv_cache: bool,
        max_text_len: int,
        trim_tail: bool,
        tail_window_size: int,
        tail_std_threshold: float,
        tail_mean_threshold: float,
        lora_stack: list | None = None,
        ref_config: dict | None = None,
        voice_design_config: dict | None = None,
        cfg_config: dict | None = None,
        rescale_config: dict | None = None,
    ):
        lora_stack = list(lora_stack or [])
        lora_paths = tuple(str(item["path"]) for item in lora_stack if item.get("path"))
        lora_scales = tuple(float(item.get("strength", 1.0)) for item in lora_stack if item.get("path"))

        runtime_key = RuntimeKey(
            checkpoint=str(model_config["checkpoint"]),
            model_device=str(model_config.get("model_device", "cuda")),
            codec_repo=str(model_config["codec_repo"]),
            model_precision=str(model_config.get("model_precision", "fp32")),
            codec_device=str(model_config.get("codec_device", "cpu")),
            codec_precision=str(model_config.get("codec_precision", "fp32")),
            enable_watermark=bool(model_config.get("enable_watermark", False)),
            compile_model=bool(model_config.get("compile_model", False)),
            compile_dynamic=bool(model_config.get("compile_dynamic", False)),
            lora_paths=lora_paths,
        )

        ref_config = ref_config or {}
        voice_design_config = voice_design_config or {}
        cfg_config = cfg_config or {}
        rescale_config = rescale_config or {}

        cfg_guidance_mode = cfg_config.get("cfg_guidance_mode", "independent")
        cfg_scale_text = float(cfg_config.get("cfg_scale_text", 3.0))
        cfg_scale_speaker = float(cfg_config.get("cfg_scale_speaker", 5.0))
        cfg_scale_override = cfg_config.get("cfg_scale_override", None)

        req = SamplingRequest(
            text=str(text),
            caption=voice_design_config.get("caption", None),
            ref_wav=ref_config.get("ref_wav", None),
            ref_latent=ref_config.get("ref_latent", None),
            no_ref=bool(ref_config.get("no_ref", ref_config.get("ref_wav", None) is None)),
            ref_normalize_db=ref_config.get("ref_normalize_db", None),
            ref_ensure_max=bool(ref_config.get("ref_ensure_max", False)),
            num_candidates=int(batch_size),
            decode_mode=str(decode_mode),
            seconds=none_if_non_positive(float(seconds)),
            duration_scale=float(duration_scale),
            min_seconds=float(min_seconds),
            max_seconds=float(max_seconds),
            max_ref_seconds=ref_config.get("max_ref_seconds", 30.0),
            max_text_len=none_if_non_positive(int(max_text_len)),
            max_caption_len=voice_design_config.get("max_caption_len", None),
            num_steps=int(num_steps),
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=float(voice_design_config.get("cfg_scale_caption", 3.0)),
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale=cfg_scale_override,
            cfg_min_t=float(cfg_config.get("cfg_min_t", 0.5)),
            cfg_max_t=float(cfg_config.get("cfg_max_t", 1.0)),
            truncation_factor=rescale_config.get("truncation_factor", None),
            rescale_k=rescale_config.get("rescale_k", None),
            rescale_sigma=rescale_config.get("rescale_sigma", None),
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=rescale_config.get("speaker_kv_scale", None),
            speaker_kv_min_t=rescale_config.get("speaker_kv_min_t", 0.9),
            speaker_kv_max_layers=rescale_config.get("speaker_kv_max_layers", None),
            seed=int(seed),
            trim_tail=bool(trim_tail),
            tail_window_size=int(tail_window_size),
            tail_std_threshold=float(tail_std_threshold),
            tail_mean_threshold=float(tail_mean_threshold),
            lora_scales=lora_scales,
        )

        runtime, _ = get_cached_runtime(runtime_key)
        progress_bar = comfy.utils.ProgressBar(int(num_steps))

        def update_progress(current: int, total: int) -> None:
            progress_bar.update_absolute(int(current), int(total))

        result = runtime.synthesize(req, log_fn=print, progress_callback=update_progress)

        audios = result.audios or [result.audio]
        max_samples = max(int(audio.shape[-1]) for audio in audios)
        padded_audios = []
        for audio in audios:
            if audio.dim() != 2:
                raise ValueError(f"Expected generated audio shape [channels, samples], got {tuple(audio.shape)}")
            if int(audio.shape[-1]) < max_samples:
                audio = torch.nn.functional.pad(audio, (0, max_samples - int(audio.shape[-1])))
            padded_audios.append(audio)

        audio_tensor = torch.stack(padded_audios, dim=0)

        out_audio = {"waveform": audio_tensor, "sample_rate": result.sample_rate}
        return io.NodeOutput(out_audio)
