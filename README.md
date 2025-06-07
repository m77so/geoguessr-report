
# GeoGuessr Report ツール

## 概要
GeoGuessrのリプレイURLからストリートビュー画像を取得し、AIによる分析・Markdownレポートを自動生成するPythonツールです。

## 必要ファイル・セットアップ

1. **Python環境**
   - Python 3.10 以上推奨
   - 依存パッケージは `pyproject.toml` で管理されています。
   - [uv](https://github.com/astral-sh/uv) などでセットアップしてください。

2. **APIキーの取得と設定**
   - Google Gemini API（GEMINI_API_KEY）
   - Google Street View Static API（STREET_API_KEY）
   - これらのAPIキーを `.env` ファイルに記載します。

#### `.env` ファイル例
```env
GEMINI_API_KEY=あなたのGemini APIキー
STREET_API_KEY=あなたのStreet View Static APIキー
```
※ `.env.example` も参考にしてください。

3. **Cookieファイル**
   - GeoGuessrのリプレイデータ取得には、ログイン済みブラウザからエクスポートした `cookies.txt` が必要です。
   - ファイル名は `cookies.txt` で、ルートディレクトリに配置してください。

## 使い方

1. 依存パッケージのインストール
   ```sh
   uv pip install -r requirements.txt  # または pyproject.toml/uv.lock を利用
   ```

2. レポート生成の実行
   ```sh
   uv run report.py --help
   # 例:
   uv run report.py --url https://www.geoguessr.com/duels/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/replay --cookie cookies.txt
   ```

3. 出力
   - 画像: `output_langchain/street_view_images_smooth/` 以下
   - Markdownレポート: `output_langchain/geoguessr_review_タイムスタンプ_ゲームID.md`

## 注意
- `.env` や `cookies.txt` には個人情報・APIキーが含まれるため、**絶対にgit管理しないでください**（`.gitignore`で除外済み）。
- APIキーは漏洩しないよう厳重に管理してください。
