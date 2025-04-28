from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import faiss
import json
import os

# 初始化 Flask
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Together.ai API设置
openai.api_key = "你的 Together.ai API Key"  # <<< 请换成你的Key
openai.base_url = "https://api.together.xyz/v1/"  # <<< Together服务器地址

# 加载症状索引
current_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(current_dir, "symptom_index.faiss")
mapping_path = os.path.join(current_dir, "symptom_mapping.json")
hospital_path = os.path.join(current_dir, "rare_disease_hospitals_top100.json")

# 加载索引
index = faiss.read_index(index_path)
with open(mapping_path, 'r', encoding='utf-8') as f:
    symptom_mapping = json.load(f)

# 加载医院知识库（带容错处理）
try:
    with open(hospital_path, 'r', encoding='utf-8') as f:
        hospital_mapping = json.load(f)
    print(f"🟢 成功加载医院知识库，共收录 {len(hospital_mapping)} 种罕见病")
except FileNotFoundError:
    print("⚠️ 警告：rare_disease_hospitals_top100.json 未找到，将禁用医院推荐功能。")
    hospital_mapping = {}

# 检索罕见病名称
def search_rare_disease(user_input):
    for disease_name in hospital_mapping.keys():
        if disease_name.split('(')[0].strip() in user_input or disease_name.split('(')[-1].strip(')') in user_input:
            return disease_name
    return None

# Flask聊天接口
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        print(f"🟠 收到用户输入: {user_message}")

        matched_disease = search_rare_disease(user_message) if hospital_mapping else None
        hospital_recommendations = []

        if matched_disease:
            hospital_recommendations = hospital_mapping.get(matched_disease, [])
            print(f"🟢 命中罕见病: {matched_disease}，推荐{len(hospital_recommendations)}家医院")

        response = openai.chat.completions.create(
            model="mistralai/Mixtral-8x7b-instruct",  # Together上优秀的模型
            messages=[
                {"role": "system", "content": "你是一个专业友好的儿童健康智能助理，帮助筛查罕见病。"},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1024
        )

        final_response = response.choices[0].message.content.strip()

        print(f"🟢 返回AI回复: {final_response}")

        return jsonify({
            'response': final_response,
            'hospitals': hospital_recommendations
        })

    except Exception as e:
        print(f"🔴 Flask处理异常: {e}")
        return jsonify({'response': "❗ 系统内部错误，请稍后再试。"}), 500

# 启动
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)