# 构建 Flask Web
# 将此单元格代码写入 predict.py 文件方便后面执行
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["POST"])  # 请求方法为 POST
def inference():
    query_df = pd.DataFrame(request.json)  # 将 JSON 变为 DataFrame
    
    df = pd.read_csv("./mushrooms.csv")  # 读取数据
    X = pd.get_dummies(df.iloc[:, 1:])  # 读取特征并独热编码
    query = pd.get_dummies(query_df).reindex(columns=X.columns, fill_value=0)  # 将请求数据 DataFrame 处理成独热编码样式
    
    clf = joblib.load('mushrooms.pkl')  # 加载模型
    prediction = clf.predict(query)  # 模型推理
    return jsonify({"prediction": list(prediction)})  # 返回推理结果
