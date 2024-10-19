from flask import Flask, jsonify, request, render_template
from model_loader import ModelLoader
from ann_recom import ann_recom, ann_recom_multi
from db import connect_db, get_item_titles_from_db_with_ids


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recom', methods=['POST'])
def get_recom():
    data = request.json
    input_info = data['titles']
    n = data.get('n')
    result = ann_recom(model_loader, connection, input_info, n)
    return jsonify(result)


@app.route('/api/recom/multi', methods=['POST'])
def get_recom_multi():
    data = request.json
    item_titles = data['titles']
    n = data.get('n')
    result = ann_recom_multi(model_loader, connection, item_titles, n)
    return jsonify(result)


@app.route('/api/recom/multi/id', methods=['POST'])
def get_recom_multi_using_id():
    data = request.json
    item_ids = data['ids']
    n = data.get('n')
    item_titles = get_item_titles_from_db_with_ids(connection,item_ids)
    result = ann_recom_multi(model_loader, connection, item_titles, n)
    return jsonify(result)


if __name__ == '__main__':
    connection = connect_db()
    model_loader = ModelLoader("train/gae_model.pth", use_gpu=False)
    app.run(debug=False, host='0.0.0.0')
