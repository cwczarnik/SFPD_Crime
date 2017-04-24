In this notebook I explore sfpd crime data and predict various crimes based on times and location.

This repository is motivated by a Kaggle competition for crime data, but ultimately I went astray and made a flask app using D3 that shows crime contours and probability of crimes based on a trained model. I trained an Xgboost model using a cross validated 70/30 split. I utilized a holdout-set of 10% for the full data set.

Prior to exploring the data I did feature engineering and one-hot-encoding the labeled crime-type feature columns (the target variables).

I explored the model efficiency and looked at the metrics. For various models and model parameters I explored the Reciever-operator curve, the F1-score, and the precision-recall curve.
