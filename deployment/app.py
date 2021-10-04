from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import flask

app = Flask(__name__)


# function to create new features
def new_feats(df):
    df2 = df.copy()
    df2['isolation_abc'] = df['isolationa'] + df['isolationb'] + df['isolationc']
    df2['isolation_def'] = df['isolationd'] + df['isolatione'] + df['isolationf']
    df2['p_IP'] = df['p0_IP'] + df['p1_IP'] + df['p2_IP']
    df2['p_p'] = df['p0_p'] + df['p1_p'] + df['p2_p']
    df2['IP_pp'] = df['IP_p0p2'] + df['IP_p1p2']
    df2['p_IPSig'] = df['p0_IPSig'] + df['p1_IPSig'] + df['p2_IPSig']
    # new feature using 'FlightDistance' and LifeTime(from literature)
    df2['FD_LT'] = df['FlightDistance'] / df['LifeTime']
    # new feature using 'FlightDistance', 'po_p', 'p1_p', 'p2_p'(from literature)
    df2['FD_p0p1p2_p'] = df['FlightDistance'] / (df['p0_p'] + df['p1_p'] + df['p2_p'])
    # new feature using 'LifeTime', 'p0_IP', 'p1_IP', 'p2_IP'(from literature)
    df2['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']) / 3
    # new feature using 'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof'(taking max value among 3 features
    # for each row)
    df2['Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    # features from kaggle discussion forum
    df2['flight_dist_sig2'] = (df['FlightDistance'] / df['FlightDistanceError']) ** 2
    df2['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']
    df2['NEW_IP_dira'] = df['IP'] * df['dira']
    df2['p0p2_ip_ratio'] = df['IP'] / df['IP_p0p2']
    df2['p1p2_ip_ratio'] = df['IP'] / df['IP_p1p2']
    df2['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df2['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df2['iso_min'] = df.loc[:,
                     ['isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf']].min(axis=1)
    return df2


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


# expected columns in the given order
columns = ['id', 'LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP', 'IPSig', 'VertexChi2', 'pt',
           'DOCAone', 'DOCAtwo', 'DOCAthree', 'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc',
           'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT',
           'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT', 'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
           'p0_IP', 'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 'p0_pt', 'p1_pt', 'p2_pt', 'p0_p',
           'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta', 'SPDhits']

# features or columns to be used for prediction
features = ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP', 'IPSig', 'VertexChi2', 'pt',
            'iso', 'ISO_SumBDT', 'isolation_abc', 'isolation_def', 'p_IP', 'p_p', 'IP_pp', 'p_IPSig', 'FD_LT',
            'FD_p0p1p2_p', 'NEW5_lt', 'Chi2Dof_MAX', 'flight_dist_sig2', 'flight_dist_sig', 'NEW_IP_dira',
            'p0p2_ip_ratio', 'p1p2_ip_ratio', 'DCA_MAX', 'iso_bdt_min', 'iso_min']


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    x = to_predict_list['input_list']
    x = x.split(",")
    y = [float(i) for i in x]
    df = pd.DataFrame(data=np.array(y).reshape(1, len(y)), columns=columns)
    df_1 = new_feats(df)
    clf = joblib.load('finalized_model.pkl')
    output = clf.predict(df_1[features])[0]
    if output == 0:
        prediction = 'background event'

    elif output == 1:
        prediction = 'signal event'

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5050)
