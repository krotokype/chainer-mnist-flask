import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from flask import Flask, request, jsonify

from model import TLP
from sys import argv
from sys import stdout

# Set up parameters
unit = 50
npz = 'model.npz'

# Set up a trained neural network
model = L.Classifier(TLP(unit, 10))
serializers.load_npz(npz, model)

# Create an application
app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    x = np.array(request.json, dtype=np.float32)
    x = x[None, ...]  # (784,) -> (1, 784)
    y = model.predictor(x)
    return jsonify(y.data[0].tolist())

app.run()
