import requests
import os
from bs4 import BeautifulSoup
import json
from PIL import Image # Pillowライブラリをインポート
import sys # エラーメッセージをsys.stderrに出力するため
import io # バイナリデータ処理用

# Pillowがインストールされていない場合は pip install Pillow を実行してください

# --- 定数設定 ---
# Street View Static API のベースURL
BASE_URL = "https://maps.googleapis.com/maps/api/streetview"

# 画像を保存するデフォルトのディレクトリ
DEFAULT_OUTPUT_DIR = "street_view_images_smooth" 

def _load_cookies_from_file(filename: str) -> str:
    """
    指定されたファイルからCookie文字列を読み込みます。

    Args:
        filename (str): Cookie文字列が保存されているファイルのパス。

    Returns:
        str: 読み込まれたCookie文字列。

    Raises:
        FileNotFoundError: Cookieファイルが見つからない場合。
        IOError: Cookieファイルの読み込み中に問題が発生した場合。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"エラー: Cookieファイル '{filename}' が見つかりません。")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            cookie_string = f.read().strip() # 末尾の改行などを除去
            if not cookie_string:
                print(f"警告: Cookieファイル '{filename}' は空です。", file=sys.stderr)
            return cookie_string
    except Exception as e:
        raise IOError(f"エラー: Cookieファイルの読み込み中に問題が発生しました: {e}")

def get_player_guesses(teams_data, player_nick):
    # teams_data = parsed_json.get('props', {}).get('pageProps', {}).get('game', {}).get('teams', [])
    for team in teams_data:
    # チーム内の全てのプレイヤーをループ
      for player in team['players']:
          print(player)
          # プレイヤーのニックネームが指定されたものと一致するかチェック
          if player.get('nick') == player_nick:
              # 一致したら、そのプレイヤーの'guesses'リストを返す
              return player.get('guesses', [])
    # 一致するプレイヤーが見つからなかった場合は空のリストを返す
    return []

def get_geoguessr_replay_data(url: str, cookie_file_path: str, player_nick: str = None) -> dict:
    """
    指定されたGeoGuessrのリプレイURLからゲームのラウンドデータをスクレイピングして返します。

    Args:
        url (str): スクレイピング対象のURL (例: 'https://www.geoguessr.com/duels/f13437be-8875-49b2-a212-3aecd9ad62b6/replay?round=1')
        cookie_file_path (str): Cookie文字列が保存されているファイルのパス。

    Returns:
        dict: 抽出されたラウンドデータの辞書（`game.rounds` の内容）。
              データが見つからない場合や空の場合は空のリストを返します。
        dict: プレイヤーの推測データ（`player_guesses`）も含まれる場合があります。

    Raises:
        FileNotFoundError: Cookieファイルが見つからない場合。
        IOError: Cookieファイルの読み込みに失敗した場合。
        requests.exceptions.RequestException: HTTPリクエスト中にエラーが発生した場合（例: 4xx, 5xx エラー）。
        json.JSONDecodeError: 取得したHTMLからJSONをパースできない場合。
        ValueError: 必要なデータ構造（`__NEXT_DATA__` スクリプトタグ、またはその中のJSONパス）が見つからない場合。
    """
    function_return_object = {
        'rounds': [],
        'player_guesses': []
    }

    print(f"URL: {url} からデータを取得中...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8',
        'Accept-Language': 'ja,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Priority': 'u=0, i',
        'TE': 'trailers'
    }

    # Cookieを読み込み、ヘッダーにセット
    # ここで _load_cookies_from_file が FileNotFoundError や IOError を発生させる可能性がある
    cookie_value = _load_cookies_from_file(cookie_file_path)
    headers['Cookie'] = cookie_value

    response = requests.get(url, headers=headers)
    response.raise_for_status() # HTTPエラーがあればrequests.exceptions.HTTPErrorを発生させる

    print("データ取得成功。HTMLをパース中...")
    soup = BeautifulSoup(response.text, 'html.parser')

    # idが__NEXT_DATA__のscriptタグを探す
    next_data_script = soup.find('script', {'id': '__NEXT_DATA__'})

    if not next_data_script:
        raise ValueError("エラー: id='__NEXT_DATA__' のスクリプトタグが見つかりませんでした。HTML構造が変わった可能性があります。")

    json_data_str = next_data_script.string
    if not json_data_str:
        raise ValueError("警告: id='__NEXT_DATA__' のスクリプトタグの中身が空です。")

    print("JSONデータをパース中...")
    parsed_json = json.loads(json_data_str)
    
    # ユーザーが指定したパスで game.rounds データを抽出
    function_return_object['rounds'] = parsed_json.get('props', {}).get('pageProps', {}).get('game', {}).get('rounds', [])

    if player_nick:
        print(f"プレイヤー '{player_nick}' の推測データを取得中...")
        player_guesses = get_player_guesses(parsed_json.get('props', {}).get('pageProps', {}).get('game', {}).get('teams', []), player_nick)
        if player_guesses:
            print(f"プレイヤー '{player_nick}' の推測データ: {len(player_guesses)} 件のデータが見つかりました。")
            # rounds_dataにプレイヤーの推測データを追加
            function_return_object['player_guesses'].extend(player_guesses)
        else:
            print(f"プレイヤー '{player_nick}' の推測データは見つかりませんでした。", file=sys.stderr)

    if not function_return_object['rounds']:
        print("警告: 抽出された game.rounds データが空または見つかりませんでした。", file=sys.stderr)

    return function_return_object
# --- 補助関数 ---

def _get_street_view_image(api_key: str, pano_id: str, size: str = "640x640", fov: int = 90, heading: int = 0, pitch: int = 0) -> bytes | None:
    """
    Street View Static APIから指定されたパノラマIDの画像をダウンロードします。

    Args:
        api_key (str): Google Cloud Street View Static APIキー。
        pano_id (str): Street ViewパノラマID。
        size (str): 画像サイズ (例: "640x640")。
        fov (int): 視野角 (Field of View)。
        heading (int): カメラの向き（北から時計回りの度数）。
        pitch (int): カメラの上下の傾き（0は水平、正は下向き、負は上向き）。

    Returns:
        bytes | None: ダウンロードされた画像データ（バイナリ）またはエラー時にNone。
    """
    params = {
        "size": size,
        "pano": pano_id,
        "fov": fov,
        "heading": heading,
        "pitch": pitch,
        "key": api_key,
    }

    # print(f"  リクエストURL: {BASE_URL}?{'&'.join([f'{k}={v}' for k,v in params.items()])}") # デバッグ用

    try:
        response = requests.get(BASE_URL, params=params, timeout=10) # タイムアウトを追加
        response.raise_for_status() 

        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            return response.content
        else:
            print(f"  エラー: パノラマID {pano_id} のレスポンスは画像ではありませんでした。Content-Type: {content_type}", file=sys.stderr)
            print(f"  レスポンステキスト (一部): {response.text[:500]}...", file=sys.stderr) 
            return None

    except requests.exceptions.Timeout:
        print(f"  APIリクエストタイムアウト (パノラマID: {pano_id})", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"  APIリクエストエラー (パノラマID: {pano_id}): {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"  HTTPステータス: {e.response.status_code}", file=sys.stderr)
            print(f"  エラー詳細: {e.response.text[:500]}...", file=sys.stderr)
        return None

def _decode_hex_pano_id(hex_string: str) -> str | None:
    """
    16進数エンコードされたStreet ViewパノラマIDを元の文字列にデコードする。
    """
    try:
        bytes_object = bytes.fromhex(hex_string)
        return bytes_object.decode('utf-8')
    except (ValueError, UnicodeDecodeError) as e:
        print(f"エラー: 無効な16進数文字列またはデコード失敗: '{hex_string}' - {e}", file=sys.stderr)
        return None

# --- メイン機能関数 ---

def save_street_view_pano_image(
    hex_pano_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    image_size: str = "640x640",
    fov: int = 90,
    pitch: int = 0,
    target_height: int = 640, # 連結後のトリミング高さ
    api_key: str = None
) -> str | None:
    """
    指定されたStreet ViewパノラマID（16進数エンコード）の画像をダウンロードし、
    複数枚を連結して指定されたディレクトリに保存します。

    Args:
        hex_pano_id (str): 16進数エンコードされたStreet ViewパノラマID。
        output_dir (str): 画像を保存するディレクトリ。デフォルトは DEFAULT_OUTPUT_DIR。
        image_size (str): ダウンロードする個々の画像のサイズ (例: "640x640")。
        fov (int): 個々の画像の視野角 (Field of View)。通常 90 または 120。
        pitch (int): カメラの上下の傾き。
        target_height (int): 連結後の画像を上から何ピクセル残すか。
        api_key (str): Google Street View Static APIキー。

    Returns:
        str | None: 保存された最終画像のファイルパス、または処理に失敗した場合はNone。
    """
    
    if api_key is None:
        api_key = os.environ.get("STREET_API_KEY")
        if not api_key:
            print("エラー: APIキーが指定されておらず、STREET_API_KEY環境変数も設定されていません。", file=sys.stderr)
            return None

    # 出力ディレクトリを作成
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"ディレクトリ '{output_dir}' を作成しました。")
        except OSError as e:
            print(f"エラー: ディレクトリ '{output_dir}' の作成に失敗しました: {e}", file=sys.stderr)
            return None

    # 16進数パノラマIDをデコード
    pano_id = _decode_hex_pano_id(hex_pano_id)
    if not pano_id:
        print(f"エラー: 16進数パノラマID '{hex_pano_id}' のデコードに失敗しました。", file=sys.stderr)
        return None

    print(f"\n処理開始: パノラマID '{pano_id}' (元の16進数: {hex_pano_id})")

    stitched_file_name = os.path.join(output_dir, f"street_view_{pano_id}_full_pano_smooth.webp")

    # ファイルが既に存在するかチェック
    if os.path.exists(stitched_file_name):
        print(f"画像 '{stitched_file_name}' は既に存在するため、ダウンロードをスキップします。")
        return stitched_file_name
    
    downloaded_image_paths = []
    
    # 360度をカバーするために必要な枚数を計算
    # fov が 360 の約数でない場合、一部が重複するか不足する可能性があるので注意
    if fov <= 0 or fov > 360:
        print(f"エラー: fov ({fov}) は1から360の範囲で指定してください。", file=sys.stderr)
        return None
    
    # 360度を完全にカバーするために、fovで割り切れない場合でも多めに取得する
    # 例: fov=120 なら 3枚、fov=90 なら 4枚
    # fov=60 なら 6枚
    num_images_needed = 360 // fov 
    # もし360度を割り切れない場合（例: fov=70）、必要に応じて次の整数に切り上げることも検討するが、
    # 今回のシナリオ(90, 120など)では整数になるため、これで十分
    headings = [d * fov for d in range(num_images_needed)]
    
    for d in headings:
        print(f"  ダウンロード中: heading={d}...")
        image_data = _get_street_view_image(api_key, pano_id, 
                                            size=image_size, 
                                            fov=fov, 
                                            heading=d, 
                                            pitch=pitch)
        if image_data:
            # 個々のダウンロード画像をWebP形式で保存
            # JPEGデータをPillowで読み込んでWebPとして再保存
            try:
                temp_image = Image.open(io.BytesIO(image_data))
                file_name = os.path.join(output_dir, f"street_view_{pano_id}_h{d:03}.webp")
                temp_image.save(file_name, format="WebP", quality=85, optimize=True)
                print(f"  画像を '{file_name}' に保存しました。")
                downloaded_image_paths.append(file_name)
            except Exception as e:
                print(f"  エラー: ファイル '{file_name}' の保存に失敗しました: {e}", file=sys.stderr)
                # エラーが発生したら、このパノラマIDの処理は中断する
                return None
        else:
            print(f"  パノラマID '{pano_id}' の heading={d} の画像の取得に失敗しました。このパノラマIDの処理を中止します。", file=sys.stderr)
            # 1枚でも失敗したら連結できないので、Noneを返す
            return None

    # すべての画像が正常に取得できた場合のみ連結処理
    if len(downloaded_image_paths) == num_images_needed:
        print(f"\nパノラマID '{pano_id}' の画像を連結します...")
        
        try:
            # 画像を読み込む (Pillowは画像ファイル名のソート順が重要なので、heading順にソートしておく)
            downloaded_image_paths.sort() 
            images = [Image.open(p) for p in downloaded_image_paths]
        except Exception as e:
            print(f"エラー: 画像ファイルの読み込みまたはソートに失敗しました: {e}", file=sys.stderr)
            return None
        
        if not images:
            print(f"  連結する画像がありません。", file=sys.stderr)
            return None
        
        first_image_width, first_image_height = images[0].size
        if not all(img.size[1] == first_image_height for img in images):
            print("警告: 取得した画像の高さが異なります。連結が正しく行われない可能性があります。", file=sys.stderr)
            # 高さが異なる場合は連結処理を中止するかどうか検討。今回はそのまま続行させるが警告。

        # 連結後の画像の幅を計算 (幅は同じであると仮定)
        total_width = sum(img.size[0] for img in images)
        max_height = first_image_height 

        # 新しい空の画像を作成
        stitched_image = Image.new('RGB', (total_width, max_height))

        # 画像を横に配置
        x_offset = 0
        for img in images:
            stitched_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        
        # 指定された高さにトリミング
        if stitched_image.height > target_height:
            stitched_image = stitched_image.crop((0, 0, stitched_image.width, target_height))
            print(f"  画像を {target_height}px の高さにトリミングしました (下 {max_height - target_height}px を破棄)。")
        else:
            print(f"  警告: 連結後の画像の高さ ({stitched_image.height}px) がトリミング目標 ({target_height}px) より小さいか同じです。トリミングはスキップされました。", file=sys.stderr)

        stitched_file_name = os.path.join(output_dir, f"street_view_{pano_id}_full_pano_smooth.webp")
        try:
            # WebP形式で品質85%、最適化して保存
            stitched_image.save(stitched_file_name, format='WebP', quality=85, optimize=True)
            print(f"  連結された画像を '{stitched_file_name}' に保存しました。")
            return stitched_file_name
        except IOError as e:
            print(f"エラー: 連結画像の '{stitched_file_name}' の保存に失敗しました: {e}", file=sys.stderr)
            return None
    else:
        print(f"  一部の画像が取得できなかったため、パノラマID '{pano_id}' の連結はスキップされました。", file=sys.stderr)
        return None

# --- スクリプトとして直接実行された場合のテストコード ---
if __name__ == "__main__":
    # !!! ここにあなたのGoogle Street View Static APIキーを貼り付けてください !!!
    # 環境変数から読み込むことを推奨します (例: os.environ.get("GOOGLE_API_KEY"))
    # API_KEY = os.environ.get("GOOGLE_API_KEY") 

    # 取得したい16進数パノラマIDのリスト
    hex_pano_ids_to_process = [
      "7475747855583341562D733979614446534C67563267", # 例1
      "7544654F773341464155726e646a4B766a7368356877", # 例2
      # 他のIDもここに追加できます
    ]

    print(f"\n--- {len(hex_pano_ids_to_process)} 個のパノラマIDの画像取得を開始します ---")

    for i, hex_id in enumerate(hex_pano_ids_to_process):
        print(f"\n--- 処理中: ({i+1}/{len(hex_pano_ids_to_process)}) ---")
        
        # 関数を呼び出す
        final_image_path = save_street_view_pano_image(hex_id)
        
        if final_image_path:
            print(f"最終画像が正常に保存されました: {final_image_path}")
        else:
            print(f"エラー: 16進数パノラマID '{hex_id}' の画像取得・保存に失敗しました。")

    print("\n--- すべての処理が完了しました ---")