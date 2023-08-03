from flask import Flask, render_template, url_for, request
from transformers import BertTokenizer
import torch
from train_test import get_prediction
from classifier import SpoilerClassifier


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    model_name = 'cointegrated/rubert-tiny'
    model = SpoilerClassifier(model_name)
    model.load_state_dict(torch.load('models/reviews_classifier'))
    tokenizer = BertTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if request.method == 'POST':
        review = request.form['review']
        tokenized_review = tokenizer.encode_plus(
                    review,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',)
        prediction = get_prediction(tokenized_review, model, device)
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0')