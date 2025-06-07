import base64
import io
import re # <= 追加
from PIL import Image, ImageDraw # <= ImageDrawを追加
import os # <= 追加
from dotenv import load_dotenv
from datetime import datetime
# LangChainとGoogleの関連クラス
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from io import BytesIO

from PIL import Image
import pycountry
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# util.py から提供された関数と定数をインポート
from util import save_street_view_pano_image, get_geoguessr_replay_rounds_data, DEFAULT_OUTPUT_DIR

load_dotenv()

# 出力ディレクトリの定義
OUTPUT_ROOT_DIR = "output_langchain"
IMAGE_SAVE_DIR = os.path.join(OUTPUT_ROOT_DIR, DEFAULT_OUTPUT_DIR)

def make_markdown_output_file(geoguessr_replay_url: str) -> str:
    """
    リプレイURLからゲームIDを抽出し、タイムスタンプと組み合わせたMarkdownファイル名を返す。
    例: output_langchain/geoguessr_review_20250607_153000_94bb58f5-53b1-47aa-9e01-81b7bca8f3bb.md
    """
    import re
    from datetime import datetime
    m = re.search(r"/([0-9a-f\-]{20,})/replay", geoguessr_replay_url)
    game_id = m.group(1) if m else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geoguessr_review_{timestamp}_{game_id}.md"
    return os.path.join(OUTPUT_ROOT_DIR, filename)

def get_country_name(country_code: str) -> str:
    """
    ISO 3166-1 alpha-2 国コードから国名を取得します。
    pycountry が国を見つけられない場合は、元の国コードを返します。
    """
    try:
        # 大文字に変換して検索
        country = pycountry.countries.get(alpha_2=country_code.upper())
        return country.name if country else country_code
    except Exception:
        # エラーが発生した場合も国コードをそのまま返す
        return country_code

def image_to_base64(pil_image: Image.Image) -> str:
    """PillowイメージをBase64エンコードされた文字列に変換します。"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def parse_and_draw_boxes(text_response: str, original_image: Image.Image, output_path: str):
    """
    LLMの応答テキストから座標をパースし、画像にバウンディングボックスを描画します。
    
    Args:
        text_response: LLMからの応答テキスト。
        original_image: 描画対象のPillowイメージオブジェクト。
        output_path: 描画後の画像を保存するパス。
    """
    # 正規表現で BOX(y_min, x_min, y_max, x_max) 形式の文字列をすべて探し出す
    # 座標値は浮動小数点数も考慮
    boxes = re.findall(r"BOX\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)", text_response)
    
    if not boxes:
        print("  描画対象のバウンディングボックスは見つかりませんでした。")
        # ボックスが見つからなくても元の画像を保存する
        original_image.save(output_path)
        return

    draw = ImageDraw.Draw(original_image)
    width, height = original_image.size

    print(f"  {len(boxes)}個のバウンディングボックスを検出しました。画像に描画します...")

    for i, box in enumerate(boxes):
        # 抽出した文字列を浮動小数点数に変換
        y_min, x_min, y_max, x_max = map(float, box)
        
        # 正規化された座標をピクセル座標に変換
        left = x_min * width
        top = y_min * height
        right = x_max * width
        bottom = y_max * height
        
        # 画像に四角形を描画
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=5)
        # 座標の近くに番号を描画して、どの言及に対応するか分かりやすくする
        draw.text((left + 5, top + 5), f"BOX {i+1}", fill="white", font_size=20)

    original_image.save(output_path)
    print(f"  バウンディングボックスを描画した画像を {output_path} に保存しました。")


def analyze_round_with_langchain_and_boxes(llm: ChatGoogleGenerativeAI, image_path: str, correct_location_str: str) -> dict:
    """
    LangChainを使い、画像分析とヒント生成を行い、言及箇所を画像に描画します。
    
    Args:
        llm: 初期化済みのChatGoogleGenerativeAIモデル。
        image_path: 分析対象の画像ファイルパス。
        correct_location_str: 正解の位置情報文字列。

    Returns:
        LLMによる予測とヒント、および注釈付き画像パスを含む辞書。
    """
    try:
        print("  LLMに場所の予測を依頼中...")
        pil_image = Image.open(image_path)
        image_b64 = image_to_base64(pil_image)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        unique_prefix = f"{base_name}_{timestamp_str}"

        chat_history = []

        # 1. 最初のメッセージを作成（場所の予測依頼 + ★座標出力の指示を追加）
        first_prompt_text = (
          "あなたはGeoGuessrの専門家です。ストリートビュー画像から場所を推測し、その根拠を画像内の特徴と関連付けて説明してください。\n\n"
          "【出力形式のルール】\n"
          "1. 推測する場所と、その理由を詳しく述べます。\n"
          "2. 画像内の具体的な特徴に言及する際は、その位置を必ず `BOX(y_min, x_min, y_max, x_max)` という形式で文中に示してください。\n"
          "3. 座標は画像の左上を(0,0)、右下を(1,1)とする正規化された値です。\n\n"
          "--- 以下は出力の例です ---\n"
          "例1:\n"
          "これは日本の田舎道だと推測します。根拠は、道路の左側に見える軽トラック `BOX(0.65, 0.20, 0.85, 0.45)` と、特徴的な形状の電柱 `BOX(0.20, 0.55, 0.70, 0.65)` です。\n\n"
          "例2:\n"
          "おそらくブラジルでしょう。ポルトガル語で書かれた看板 `BOX(0.40, 0.60, 0.55, 0.80)` があります。また、独特の赤土 `BOX(0.85, 0.0, 1.0, 1.0)` も特徴的です。\n"
          "--- 例はここまで ---\n\n"
          "それでは、この画像について分析を開始してください。"
        )
        first_message = HumanMessage(
            content=[
                {"type": "text", "text": first_prompt_text},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
            ]
        )
        chat_history.append(first_message)
        
        response1 = llm.invoke(chat_history)
        llm_prediction = response1.content
        print(response1)
        chat_history.append(response1)

        print(f"  LLMの予測 (一部): {llm_prediction[:150]}...")
        # ★予測の応答に基づいて画像に描画

        annotated_prediction_path = os.path.join(IMAGE_SAVE_DIR, f"{unique_prefix}_prediction.png")
        parse_and_draw_boxes(llm_prediction, pil_image.copy(), annotated_prediction_path)

        # 2. ２回目のメッセージを作成（正解を伝え、ヒントを依頼 + ★座標出力の指示を追加）
        print("  LLMに正解を伝え、追加のヒントを依頼中...")
        second_prompt_text = (
            f"ありがとう。実は、この場所の正確な位置は {correct_location_str} でした。"
            f"あなたの先ほどの予測と比べてどうでしたか？"
            f"この場所特有のGeoGuessrで使えるヒントや、今後の推測に役立つ特徴（道路標識、植生、建築様式、電柱、車両、ナンバープレートなど）があれば具体的に教えてください。"
            f"特に、画像から読み取れる視覚的な手掛かりを優先し、その場所を BOX(y_min, x_min, y_max, x_max) 形式で示してください。"
        )
        second_message = HumanMessage(content=second_prompt_text)
        chat_history.append(second_message)

        response2 = llm.invoke(chat_history)
        llm_hint = response2.content
        print(f"  LLMの追加ヒント (一部): {llm_hint[:150]}...")
        # ★ヒントの応答に基づいて画像に描画
        annotated_hint_path =  os.path.join(IMAGE_SAVE_DIR,f"{unique_prefix}_annotated_hint.png")
        parse_and_draw_boxes(llm_hint, pil_image.copy(), annotated_hint_path)


        return {
            "llm_prediction": llm_prediction,
            "annotated_prediction_path": annotated_prediction_path,
            "llm_hint": llm_hint,
            "annotated_hint_path": annotated_hint_path,
            "error": None
        }

    except Exception as e:
        print(f"  LLMとの対話中にエラーが発生しました: {e}")
        return {
            "llm_prediction": "エラーにより取得できませんでした。",
            "annotated_prediction_path": None,
            "llm_hint": "エラーにより取得できませんでした。",
            "annotated_hint_path": None,
            "error": f"{type(e).__name__}: {e}"
        }


def main(geoguessr_replay_url,cookie_file_path ):

    # 環境変数からAPIキーを取得
    GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
    STREET_API_KEY = os.environ.get("STREET_API_KEY")

    if not GOOGLE_API_KEY:
        print("エラー: GEMINI_API_KEY環境変数が設定されていません。")
        print("Google CloudコンソールでAPIキーを作成し、環境変数に設定するか、.envファイルに記述してください。")
        return

    if not STREET_API_KEY:
        print("エラー: STREET_API_KEY環境変数が設定されていません。")
        print("Google CloudコンソールでAPIキーを作成し、環境変数に設定するか、.envファイルに記述してください。")
        return

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", google_api_key=GOOGLE_API_KEY)
        print("LangChain経由でGemini 2.5 Flash モデルを初期化しました。\n")
    except Exception as e:
        print(f"モデルの初期化に失敗しました: {e}")
        return

    # Markdownファイル名を生成
    markdown_output_file = make_markdown_output_file(geoguessr_replay_url)
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(markdown_output_file), exist_ok=True)

    print("\n--- GeoGuessr リプレイデータ取得 ---")
    try:
        rounds_data = get_geoguessr_replay_rounds_data(geoguessr_replay_url, cookie_file_path)
        if not rounds_data:
            print("リプレイデータが見つかりませんでした。URLまたはCookieファイルを確認してください。")
            return
        print(f"合計 {len(rounds_data)} ラウンドのデータが見つかりました。")
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("Cookieファイルのパスが正しいか確認してください。")
        return
    except Exception as e:
        print(f"リプレイデータの取得中に予期せぬエラーが発生しました: {e}")
        return

    review_results = []

    for i, round_data in enumerate(rounds_data):
        round_number = round_data.get("roundNumber", i + 1)
        panorama = round_data.get("panorama", {})
        pano_id = panorama.get("panoId")
        lat = panorama.get("lat")
        lng = panorama.get("lng")
        country_code = panorama.get("countryCode")

        # 必要なデータが揃っているかチェック
        if not all([pano_id, lat is not None, lng is not None, country_code]):
            print(f"--- ラウンド {round_number} ---")
            print("  必要なデータ (panoId, lat, lng, countryCode) が不完全なため、スキップします。")
            review_results.append({
                "round_number": round_number,
                "status": "データ不完全",
                "correct_location": "N/A",
                "image_path": "N/A",
                "llm_prediction": "N/A",
                "llm_hint": "N/A"
            })
            continue

        print(f"\n--- ラウンド {round_number} の処理を開始 ---")
        correct_country_name = get_country_name(country_code)
        correct_location_str = f"{correct_country_name} (Lat: {lat:.4f}, Lng: {lng:.4f})"
        # 1. 画像のダウンロード
        print(f"  パノラマID '{pano_id}' の画像をダウンロード中...")
        image_path = save_street_view_pano_image(pano_id, output_dir=IMAGE_SAVE_DIR, api_key=STREET_API_KEY)

        if not image_path or not os.path.exists(image_path):
            print(f"  エラー: ラウンド {round_number} の画像ダウンロードに失敗しました。スキップします。")
            review_results.append({
                "round_number": round_number,
                "status": "画像ダウンロード失敗",
                "correct_location": correct_location_str,
                "image_path": "N/A",
                "llm_prediction": "N/A",
                "llm_hint": "N/A"
            })
            continue
        print(f"  画像が '{image_path}' に保存されました。")

        # 2. LangChainを使って画像分析
        analysis = analyze_round_with_langchain_and_boxes(llm, image_path, correct_location_str)

        # 3. 結果をまとめる
        status = "成功" if analysis["error"] is None else f"LLMエラー: {analysis['error']}"
        review_results.append({
            "round_number": round_number,
            "status": status,
            "correct_location": correct_location_str,
            "image_path": image_path,
            **analysis # llm_predictionとllm_hintを展開して追加
        })

    # 結果をMarkdown形式で出力
    print(f"\n--- レビュー結果を '{markdown_output_file}' に出力中 ---")
    with open(markdown_output_file, "w", encoding="utf-8") as f:
        f.write("# GeoGuessr 反省レポート\n\n")
        f.write(f"**リプレイURL**: {geoguessr_replay_url}\n\n")
        f.write("---\n\n")

        if not review_results:
            f.write("処理されたラウンドデータがありませんでした。\n\n")
        else:
            for result in review_results:
                f.write(f"## ラウンド {result['round_number']}\n\n")
                f.write(f"**ステータス**: {result['status']}\n\n")
                f.write(f"**正解の場所**: {result['correct_location']}\n\n")
                
                # ----------------------------------------------------------------------
                # 元のストリートビュー画像
                # ----------------------------------------------------------------------
                f.write(f"### ストリートビュー画像\n")
                if result.get('image_path') != "N/A" and result.get('image_path') and os.path.exists(result['image_path']):
                    # Markdownファイルからの相対パスを計算
                    relative_image_path = os.path.relpath(result['image_path'], start=os.path.dirname(markdown_output_file))
                    f.write(f"![ラウンド {result['round_number']} ストリートビュー]({relative_image_path.replace(os.path.sep, '/')})\n\n") # パス区切り文字を / に統一
                else:
                    f.write("画像は利用できませんでした。\n\n")

                # ----------------------------------------------------------------------
                # LLMの予測と、それに対応する注釈付き画像
                # ----------------------------------------------------------------------
                f.write(f"### LLMの予測\n")
                f.write(f"{result.get('llm_prediction', '予測テキストはありません。')}\n\n")
                
                # 予測の注釈付き画像があるかチェック
                annotated_pred_path = result.get('annotated_prediction_path')
                if annotated_pred_path and os.path.exists(annotated_pred_path):
                    f.write(f"#### LLMが注目した箇所 (予測時)\n")
                    relative_pred_img_path = os.path.relpath(annotated_pred_path, start=os.path.dirname(markdown_output_file))
                    f.write(f"![予測の注釈付き画像]({relative_pred_img_path.replace(os.path.sep, '/')})\n\n")
                
                # ----------------------------------------------------------------------
                # LLMのヒントと、それに対応する注釈付き画像
                # ----------------------------------------------------------------------
                f.write(f"### LLMからの追加ヒント\n")
                f.write(f"{result.get('llm_hint', 'ヒントテキストはありません。')}\n\n")

                # ヒントの注釈付き画像があるかチェック
                annotated_hint_path = result.get('annotated_hint_path')
                if annotated_hint_path and os.path.exists(annotated_hint_path):
                    f.write(f"#### LLMが注目した箇所 (ヒント時)\n")
                    relative_hint_img_path = os.path.relpath(annotated_hint_path, start=os.path.dirname(markdown_output_file))
                    f.write(f"![ヒントの注釈付き画像]({relative_hint_img_path.replace(os.path.sep, '/')})\n\n")
                    
                f.write("---\n\n")

    print(f"処理が完了しました。結果は '{markdown_output_file}' を参照してください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoGuessrリプレイURLとCookieファイルを指定してレポートを生成します。")
    parser.add_argument("--url", required=True, help="GeoGuessrのリプレイURL (例: https://www.geoguessr.com/duels/xxxx/replay)")
    parser.add_argument("--cookie", required=True, help="GeoGuessrのCookieファイルパス (例: cookies.txt)")
    args = parser.parse_args()
    main(args.url, args.cookie)