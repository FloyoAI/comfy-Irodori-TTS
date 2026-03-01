# ComfyUI_IrodoriTTS_Wrapper

[IrodoriTTS](https://github.com/Aratako/Irodori-TTS)のComfyUI用カスタムノードです。

## install
1. custom_nodesフォルダ上でgit clone
2. 仮想環境を有効化した上で `pip install -r ComfyUI_IrodoriTTS_Wrapper/requirements.txt`
3. 必要ならcheckpointsフォルダに[IrodoriTTSモデル](https://huggingface.co/Aratako/Irodori-TTS-500M/blob/main/model.safetensors)をDLする


## ノード一覧
- IrodoriTTS Model Loader
- IrodoriTTS Model Loader HF
  - モデルを読み込みます
  - HFの方ははhuggingfaceリポジトリからDLします

- IrodoriTTSSampler
  - テキストの入力と実際の生成を行います

- IrodoriTTS Referenec Audio
- IrodoriTTS Advanced CFG
- IrodoriTTS Rescale Config
  - オプション設定用のカスタムノードです

- IrodoriTTS Emoji Selector
  - IrodoriTTSで使用できる絵文字一覧です。
  - ボタンクリックで、クリップボードに絵文字をコピーします
