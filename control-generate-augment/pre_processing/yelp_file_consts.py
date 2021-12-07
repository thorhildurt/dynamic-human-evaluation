import pathlib

# The source data folder, ignored in git.
data_dir = pathlib.Path('..', 'data', 'yelp')
yelp_train = "yelp_train.csv"
yelp_valid = "yelp_valid.csv"
yelp_test = "yelp_test.csv"

y_yelp_train = "y_yelp_train.csv"
y_yelp_valid = "y_yelp_valid.csv"
y_yelp_test = "y_yelp_test.csv"

proxy_yelp_train = "proxy_yelp_train.csv"
proxy_y_yelp_train = "proxy_y_yelp_train.csv"

y_yelp_train_complete = "y_yelp_train_complete"
y_yelp_valid_complete = "y_yelp_valid_complete"
y_yelp_test_complete = "y_yelp_test_complete"

tense_data_dir = pathlib.Path(data_dir, 'tense_data')
pron_data_dir = pathlib.Path(data_dir, 'pronoun_data')
