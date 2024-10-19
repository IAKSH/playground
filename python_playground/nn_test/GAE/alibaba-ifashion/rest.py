from flask import Flask, jsonify, request, render_template
from model_loader import ModelLoader
from ann_recom import ann_recom, ann_recom_multi


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recom', methods=['POST'])
def get_recom():
    data = request.json
    input_info = data['info']
    n = data.get('n', 5)
    result = ann_recom(model_loader, input_info, n)
    return jsonify(result)


@app.route('/api/recom_multi', methods=['POST'])
def get_recom_multi():
    data = request.json
    item_titles = data['item_titles']
    n = data.get('n', 5)
    result = ann_recom_multi(model_loader, item_titles, n)
    return jsonify(result)


if __name__ == '__main__':
    model_loader = ModelLoader("gae_model.pth")
    app.run(debug=False, host='0.0.0.0')
