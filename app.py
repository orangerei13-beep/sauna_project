# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- 0. トップページ追加 ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')  # index.html を返す

# --- 1. サウナデータの読み込みとAI準備 ---
try:
    file_path = os.path.join(os.path.dirname(__file__), '礼 サウナ 全国版.xlsx - サウナ.csv')
    df = pd.read_csv(file_path)
    def clean_price(price):
        if isinstance(price, str):
            num_str = re.search(r'(\d{1,3}(,\d{3})*|\d+)', price)
            if num_str:
                return int(num_str.group().replace(',', ''))
        return None
    df['料金_数値'] = df['料金'].apply(clean_price)
    required_cols = ['初心者におすすめのポイント', 'スッキリの種類', 'サウナの温度', '水風呂の温度']
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['features'] = df['初心者におすすめのポイント'] + " " + \
                     df['スッキリの種類'] + " " + \
                     df['サウナの温度'] + " " + \
                     df['水風呂の温度']
    tfidf_vectorizer = TfidfVectorizer(stop_words=['て', 'に', 'を', 'は', 'が', 'です', 'ます'])
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['features'])
    print(":チェックマーク_緑: AIモデル準備完了 (新ロジック)")
except FileNotFoundError:
    print(":x: エラー: CSVファイルが見つかりません。")
    df = None

# --- 2. 投稿データの管理機能 ---
POSTS_FILE = 'posts.json'
def load_posts():
    if not os.path.exists(POSTS_FILE): return []
    try:
        with open(POSTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_post_data(new_post):
    posts = load_posts()
    posts.insert(0, new_post)
    with open(POSTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False, indent=4)

# --- 3. APIエンドポイント ---
@app.route('/recommend', methods=['POST'])
def recommend():
    if df is None: return jsonify({"error": "データなし"}), 500
    answers = request.json
    print(f":デスクトップコンピューター: ブラウザから受け取った回答: {answers}")
    user_preference_text = f"{answers.get('refresh_type', '')} {answers.get('sauna_temp', '')} {answers.get('water_temp', '')}"
    user_vec = tfidf_vectorizer.transform([user_preference_text])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    recommendations = []
    for i in related_docs_indices:
        sauna = df.iloc[i]
        recommendations.append({
            '施設名': sauna['施設名'],
            '場所': sauna['場所'],
            '料金': sauna['料金'],
            '初心者におすすめのポイント': sauna['初心者におすすめのポイント']
        })
    return jsonify(recommendations)

@app.route('/posts', methods=['GET'])
def get_posts():
    return jsonify(load_posts())

@app.route('/posts', methods=['POST'])
def add_post():
    data = request.json
    if not data.get('name') or not data.get('content'):
        return jsonify({"error": "名前と内容は必須です"}), 400
    data['date'] = datetime.now().strftime('%Y/%m/%d %H:%M')
    save_post_data(data)
    return jsonify({"message": "投稿しました！", "post": data})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
