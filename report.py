import base64
import io
import re
from PIL import Image, ImageDraw, ImageFont
import os
import concurrent.futures
from dotenv import load_dotenv
from datetime import datetime
import argparse
from io import BytesIO
import requests # 追加

import pycountry
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

# util.py から提供された関数と定数をインポート
from util import save_street_view_pano_image, get_geoguessr_replay_data, DEFAULT_OUTPUT_DIR

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

def get_address_from_coords(lat: float, lng: float, api_key: str, language: str = "ja") -> str | None:
    """
    緯度経度からGoogle Geocoding APIを使用して住所を取得します。
    """
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}&language={language}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        data = response.json()
        if data.get("status") == "OK" and data.get("results"):
            # 複数の結果から最も適切と思われるものを選択（通常は最初のもの）
            return data["results"][0].get("formatted_address")
        else:
            print(f"  Geocoding APIから有効な結果が得られませんでした: {data.get('status')}, {data.get('error_message')}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  Geocoding APIへのリクエスト中にエラーが発生しました: {e}")
        return None
    except Exception as e:
        print(f"  住所の解析中に予期せぬエラーが発生しました: {e}")
        return None

def image_to_base64(pil_image: Image.Image) -> str:
    """PillowイメージをBase64エンコードされた文字列に変換します。"""
    buffered = io.BytesIO()
    # WebP形式で高圧縮率で保存（品質85%、ロスレス圧縮も試す場合はlossless=True）
    pil_image.save(buffered, format="WebP", quality=85, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def parse_and_draw_boxes(text_response: str, original_image: Image.Image, output_path: str):
    """
    LLMの応答テキストから座標をパースし、画像にバウンディングボックスを描画します。
    
    Args:
        text_response: LLMからの応答テキスト。
        original_image: 描画対象のPillowイメージオブジェクト。
        output_path: 描画後の画像を保存するパス。
    """    # 正規表現で BOX(y_min, x_min, y_max, x_max) 形式の文字列をすべて探し出す
    # 座標値は浮動小数点数も考慮
    boxes = re.findall(r"BOX\((\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\)", text_response)
    
    if not boxes:
        print("  描画対象のバウンディングボックスは見つかりませんでした。")
        # ボックスが見つからなくても元の画像を保存する（WebP高圧縮）
        original_image.save(output_path, format="WebP", quality=85, optimize=True)
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

    # WebP高圧縮で保存
    original_image.save(output_path, format="WebP", quality=85, optimize=True)
    print(f"  バウンディングボックスを描画した画像を {output_path} に保存しました。")


def analyze_round_with_langchain_and_boxes(llm: BaseChatModel, image_path: str, correct_location_str: str) -> dict:
    """
    LangChainを使い、画像分析とヒント生成を行い、言及箇所を画像に描画します。
    Args:
        llm: 初期化済みのBaseChatModelモデル。
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
          "あなたはGeoGuessrの専門家です。ストリートビュー画像から場所を推測し、その根拠を画像内の特徴と関連付けて説明してください。類似している地域を否定する理由も述べてください\n\n"
          "添付される画像はGeoGuessrのストリートビュー画像です。左から、北、東、南、西を向いた画像を張り合わせたものです。\n\n"
          "【出力形式のルール】\n"
          "1. 推測する場所と、その理由を詳しく述べます。\n"
          "2. 画像内の具体的な特徴に言及する際は、その位置を必ず `BOX(y_min, x_min, y_max, x_max)` という形式で文中に示してください。\n"
          "3. 座標は画像の左上を(0,0)、右下を(1,1)とする正規化された値です。\n"
          "4. 文字数は全体で3000文字以内にしてください\n\n"
          "--- 以下は出力の例です ---\n"
          "例1:\n"
          "これは日本の田舎道だと推測します。根拠は、道路の左側に見える軽トラック `BOX(0.65, 0.20, 0.85, 0.45)` と、特徴的な形状の電柱 `BOX(0.20, 0.55, 0.70, 0.65)` です。光沢のある黒い瓦の色から、石川県と推測されます。\n\n"
          "例2:\n"
          "おそらくブラジルでしょう。ポルトガル語で書かれた看板 `BOX(0.40, 0.60, 0.55, 0.80)` があります。また、独特の赤土 `BOX(0.85, 0.0, 1.0, 1.0)` も特徴的です。\n"
          "--- 例はここまで ---\n\n"
          "それでは、この画像について分析を開始してください。"
        )
        first_message = HumanMessage(
            content=[
                {"type": "text", "text": first_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                },
            ]
        )
        chat_history.append(first_message)
        
        response1 = llm.invoke(chat_history)
        llm_prediction = response1.content
        print(response1)
        chat_history.append(response1)

        print(f"  LLMの予測 (一部): {llm_prediction[:150]}...")

        annotated_prediction_path = os.path.join(IMAGE_SAVE_DIR, f"{unique_prefix}_prediction.webp")
        parse_and_draw_boxes(llm_prediction, pil_image.copy(), annotated_prediction_path)

        # 2. ２回目のメッセージを作成（正解を伝え、ヒントを依頼 + 座標出力の指示を追加）
        print("  LLMに正解を伝え、追加のヒントを依頼中...")
        second_prompt_text = (
            f"ありがとう。実は、この場所の正確な位置は {correct_location_str} でした。"
            f"あなたの先ほどの予測と比べてどうでしたか？また、プレイヤーの推測に対しても批評してください。"
            f"この場所特有のGeoGuessrで使えるヒントや、今後の推測に役立つ特徴（道路標識、植生、建築様式、電柱、車両、ナンバープレートなど）があれば具体的に教えてください。"
            f"特に、画像から読み取れる視覚的な手掛かりを優先し、その場所を BOX(y_min, x_min, y_max, x_max) 形式で示してください。"
        )
        second_message = HumanMessage(content=second_prompt_text)
        chat_history.append(second_message)

        response2 = llm.invoke(chat_history)
        llm_hint = response2.content
        print(f"  LLMの追加ヒント (一部): {llm_hint[:150]}...")
        annotated_hint_path =  os.path.join(IMAGE_SAVE_DIR,f"{unique_prefix}_annotated_hint.webp")
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



def main(geoguessr_replay_url,cookie_file_path, model_name="gemini-2.5-flash-preview-05-20", rounds_to_process_str=None):

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

    # モデル名のプレフィックスで分岐
    if model_name.startswith("gemini"):
        print(f"Geminiモデル ({model_name}) を使用します。")
        GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            print("エラー: GEMINI_API_KEY または GOOGLE_API_KEY 環境変数が設定されていません。")
            return
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.5,
            )
        except Exception as e:
            print(f"Geminiモデルの初期化に失敗しました: {e}")
            return
            
    elif model_name.startswith("gpt") or model_name.startswith("o"):
        print(f"OpenAIモデル ({model_name}) を使用します。")
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            print("エラー: OPENAI_API_KEY環境変数が設定されていません。")
            return
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=OPENAI_API_KEY,
                temperature=1.0 if model_name.startswith("o") else 0.5
            )
        except Exception as e:
            print(f"OpenAIモデルの初期化に失敗しました: {e}")
            return
    else:
        print(f"エラー: サポートされていないモデル名です: {model_name}")
        print("モデル名は 'gemini-' または 'gpt-' で始まる必要があります。")
        return
    
    print(f"LangChain経由で {model_name} モデルを正常に初期化しました。\n")

    # Markdownファイル名を生成
    markdown_output_file = make_markdown_output_file(geoguessr_replay_url)
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(markdown_output_file), exist_ok=True)

    print("\n--- GeoGuessr リプレイデータ取得 ---")
    try:
        replay_data = get_geoguessr_replay_data(geoguessr_replay_url, cookie_file_path, player_nick=args.player_nickname)
        if not replay_data['rounds']:
            print("リプレイデータが見つかりませんでした。URLまたはCookieファイルを確認してください。")
            return
        print(f"合計 {len(replay_data['rounds'])} ラウンドのデータが見つかりました。")
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("Cookieファイルのパスが正しいか確認してください。")
        return
    except Exception as e:
        print(f"リプレイデータの取得中に予期せぬエラーが発生しました: {e}")
        return

    # 指定されたラウンドのみを処理するためのフィルタリングロジック
    if rounds_to_process_str:
        try:
            # カンマで分割し、空白を除去し、整数に変換してセットを作成
            specified_rounds = {int(r.strip()) for r in rounds_to_process_str.split(',')}
            print(f"指定されたラウンド {sorted(list(specified_rounds))} のみを処理します。")
            
            # replay_data['rounds']をフィルタリング
            original_rounds_data = replay_data['rounds']
            rounds_to_process = [
                round_data for round_data in original_rounds_data
                if round_data.get("roundNumber") in specified_rounds
            ]
            
            if not rounds_to_process:
                print("指定されたラウンド番号に一致するデータが見つかりませんでした。処理を終了します。")
                return
            
            # 処理対象のラウンドリストを上書き
            replay_data['rounds'] = rounds_to_process
            print(f"フィルタリング後の処理対象ラウンド数: {len(replay_data['rounds'])}")

        except ValueError:
            print("エラー: --rounds の指定が正しくありません。ラウンド番号をカンマ区切りで指定してください (例: 1,3,5)。")
            return
    # --- フィルタリングロジックここまで ---

    review_results = []

    # Helper function to process a single round
    def process_round_task(args_tuple):
        i, round_data, llm_model, street_api_key_val, replay_player_guesses, image_save_dir_val = args_tuple
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
            return {
                "round_number": round_number,
                "status": "データ不完全",
                "correct_location": "N/A",
                "image_path": "N/A",
                "llm_prediction": "N/A",
                "llm_hint": "N/A",
                "error": "Incomplete data" # Add error field for consistency
            }

        # This print might interleave in parallel execution, consider logging or removing for cleaner output
        print(f"--- ラウンド {round_number} の処理を開始 ---")
        correct_country_name = get_country_name(country_code)
        
        # Geocoding APIを使用して住所を取得
        formatted_address = get_address_from_coords(lat, lng, street_api_key_val)
        
        if formatted_address:
            correct_location_str = f"住所: {formatted_address} (国: {correct_country_name}, 緯度: {lat:.4f}, 経度: {lng:.4f})"
        else:
            # 住所取得に失敗した場合は従来通りの形式
            correct_location_str = f"{correct_country_name} (緯度: {lat:.4f}, 経度: {lng:.4f})"


        if replay_player_guesses:
            player_guess = next((g for g in replay_player_guesses if g.get("roundNumber") == round_number), None)
            if player_guess:
                guess_lat = player_guess.get("lat")
                guess_lng = player_guess.get("lng")
                if guess_lat is not None and guess_lng is not None:
                    formatted_address = get_address_from_coords(guess_lat, guess_lng, street_api_key_val)
                    if formatted_address:
                        correct_location_str += f" 、プレイヤーの推測は: {formatted_address} （Lat: {guess_lat:.4f}, Lng: {guess_lng:.4f})"
                    else:
                        correct_location_str += f" 、プレイヤーの推測は: （Lat: {guess_lat:.4f}, Lng: {guess_lng:.4f})"
                else:
                    # This print might interleave
                    print(f"  ラウンド {round_number} のプレイヤーの推測が不完全なため、正解の場所のみを表示します。")
        
        # 1. 画像のダウンロード
        # This print might interleave
        print(f"  ラウンド {round_number}: パノラマID '{pano_id}' の画像をダウンロード中...")
        image_path = save_street_view_pano_image(pano_id, output_dir=image_save_dir_val, api_key=street_api_key_val)

        if not image_path or not os.path.exists(image_path):
            # This print might interleave
            print(f"  エラー: ラウンド {round_number} の画像ダウンロードに失敗しました。スキップします。")
            return {
                "round_number": round_number,
                "status": "画像ダウンロード失敗",
                "correct_location": correct_location_str,
                "image_path": "N/A",
                "llm_prediction": "N/A",
                "llm_hint": "N/A",
                "error": "Image download failed" # Add error field
            }
        # This print might interleave
        print(f"  ラウンド {round_number}: 画像が '{image_path}' に保存されました。")

        # 2. LangChainを使って画像分析
        analysis = analyze_round_with_langchain_and_boxes(llm_model, image_path, correct_location_str)

        # 3. 結果をまとめる
        status = "成功" if analysis["error"] is None else f"LLMエラー: {analysis['error']}"
        return {
            "round_number": round_number,
            "status": status,
            "correct_location": correct_location_str,
            "image_path": image_path,
            **analysis # llm_predictionとllm_hintを展開して追加
        }

    # Prepare arguments for each task
    tasks_args = []
    for i, round_data_item in enumerate(replay_data['rounds']):
        tasks_args.append((i, round_data_item, llm, STREET_API_KEY, replay_data.get("player_guesses"), IMAGE_SAVE_DIR))

    # Use ThreadPoolExecutor to process rounds in parallel
    # Adjust max_workers based on your system and API rate limits
    # Using a small number like 5 to avoid overwhelming the LLM API or local resources
    # If the LLM API has strict rate limits, a lower number or sequential processing might still be necessary.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Using map to preserve order, though sorting later anyway
        # review_results_unordered = list(executor.map(process_round_task, tasks_args))
        
        # Using submit and as_completed for potentially faster perceived progress if some tasks are slow
        future_to_round_args = {executor.submit(process_round_task, task_arg): task_arg for task_arg in tasks_args}
        review_results_unordered = []
        for future in concurrent.futures.as_completed(future_to_round_args):
            try:
                data = future.result()
                review_results_unordered.append(data)
            except Exception as exc:
                # This part handles exceptions raised directly by the task function if not caught inside
                # For example, if process_round_task itself had an unhandled exception
                # However, process_round_task is designed to catch its own errors and return a dict
                original_task_args = future_to_round_args[future]
                round_idx = original_task_args[0]
                round_num_fallback = replay_data['rounds'][round_idx].get("roundNumber", round_idx + 1)
                print(f'ラウンド {round_num_fallback} の処理中にエラーが発生しました: {exc}')
                review_results_unordered.append({
                    "round_number": round_num_fallback,
                    "status": f"並列処理エラー: {exc}",
                    "correct_location": "N/A",
                    "image_path": "N/A",
                    "llm_prediction": "N/A",
                    "llm_hint": "N/A",
                    "error": str(exc)
                })


    # Sort results by round_number to ensure the report is in order
    review_results = sorted(review_results_unordered, key=lambda x: x.get("round_number", float('inf')))


    # 結果をMarkdown形式で出力
    print(f"\\n--- レビュー結果を '{markdown_output_file}' に出力中 ---")
    with open(markdown_output_file, "w", encoding="utf-8") as f:
        f.write("# GeoGuessr 振り返りレポート\n\n")
        f.write(f"**リプレイURL**: {geoguessr_replay_url}\n\n**担当モデル**: {model_name}\n")
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
    parser.add_argument(
        "--model", 
        default="gemini-2.5-flash-preview-05-20", 
        help="使用するLLMモデル名。'gemini-'か'gpt-'で始まる名前を指定 (例: gemini-1.5-flash, gpt-4o)。(デフォルト: gemini-2.5-flash-preview-05-20)"
    )
    parser.add_argument("--player_nickname", default="LangChain", help="プレイヤーのニックネーム (デフォルト: LangChain)")
    parser.add_argument(
        "--rounds",
        type=str,
        default=None,
        help="処理するラウンド番号をカンマ区切りで指定します (例: 2,4)。指定しない場合は全ラウンドを処理します。"
    )

    
    args = parser.parse_args()
    main(args.url, args.cookie, args.model, args.rounds)