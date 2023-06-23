import time
import pickle
import os
import gc

import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split

def add_date_features(
    in_data, datetime_key="date_time", features=["month", "hour", "dayofweek"]
):
    dates = pd.to_datetime(in_data[datetime_key])
    for feature in features:
        if feature == "month":
            in_data["month"] = dates.dt.month
        elif feature == "dayofweek":
            in_data["dayofweek"] = dates.dt.dayofweek
        elif feature == "hour":
            in_data["hour"] = dates.dt.hour

    return in_data

def drop_columns_with_missing_data(
    df,
    threshold,
    ignore_values=[
        "visitor_hist_adr_usd",
        "visitor_hist_starrating",
        "srch_query_affinity_score",
    ],
):
    columns_to_drop = []

    for i in range(df.shape[1]):
        length_df = len(df)
        column_names = df.columns.tolist()
        number_nans = sum(df.iloc[:, i].isnull())
        if number_nans / length_df > threshold:
            if column_names[i] not in ignore_values:
                columns_to_drop.append(column_names[i])

    print(
        "Dropping columns {} because they miss more than {} of data.".format(
            columns_to_drop, threshold
        )
    )

    df_reduced = df.drop(labels=columns_to_drop, axis=1)
    print("Dropped columns {}".format(columns_to_drop))
    return df_reduced


def remove_columns(x1, ignore_column=["srch_id", "prop_id", "position", "random_bool"]):
    ignore_column = [c for c in ignore_column if c in x1.columns.values]
    x1 = x1.drop(labels=ignore_column, axis=1)
    return x1


def input_estimated_position(training_data, srch_id_dest_id_dict):
    training_data = training_data.merge(
        srch_id_dest_id_dict, how="left", on=["srch_destination_id", "prop_id"]
    )
    print(training_data.head())
    return training_data

def get_categorical_column(x1):
    categorical_features = [
        "day",
        "month",
        "prop_country_id",
        "site_id",
        "visitor_location_country_id",
    ]
    categorical_features = [c for c in categorical_features if c in x1.columns.values]
    categorical_features_numbers = [x1.columns.get_loc(x) for x in categorical_features]
    return categorical_features_numbers


def train_model(
    x1, x2, y1, y2, groups, eval_groups, lr, method, output_dir):

    categorical_features_numbers = get_categorical_column(x1)
    clf = lightgbm.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=5000,
        learning_rate=lr,
        max_position=5,
        label_gain=[0, 1, 5],
        random_state=69,
        seed=69,
        boosting=method,
    )
    gc.collect()

    print("Training on train set with columns: {}".format(x1.columns.values))
    clf.fit(
        x1,
        y1,
        eval_set=[(x1, y1), (x2, y2)],
        eval_group=[groups, eval_groups],
        group=groups,
        eval_at=5,
        verbose=20,
        early_stopping_rounds=200,
        categorical_feature=categorical_features_numbers,
    )
    gc.collect()
    pickle.dump(clf, open(os.path.join(output_dir, "model.dat"), "wb"))
    return clf


def predict(test_data, srch_id_dest_id_dict, output_dir):

    gc.collect()

    model = pickle.load(open(os.path.join(output_dir, "model.dat"), "rb"))

    test_data = test_data.copy()
    test_data = input_estimated_position(test_data, srch_id_dest_id_dict)

    test_data_srch_id_prop_id = test_data[["srch_id", "prop_id"]]

    test_data = remove_columns(test_data)

    categorical_features_numbers = get_categorical_column(test_data)

    print("Predicting on train set with columns: {}".format(test_data.columns.values))
    kwargs = {}
    kwargs = {"categorical_feature": categorical_features_numbers}

    predictions = model.predict(test_data, **kwargs)
    test_data_srch_id_prop_id["prediction"] = predictions
    del test_data
    gc.collect()

    test_data_srch_id_prop_id = test_data_srch_id_prop_id.sort_values(
        ["srch_id", "prediction"], ascending=False
    )
    print("Saving predictions into submission.csv")
    test_data_srch_id_prop_id[["srch_id", "prop_id"]].to_csv(
        os.path.join(output_dir, "submission.csv"), index=False
    )


def preprocess_training_data(orig_data, kind="train", use_ndcg_choices=False):

    print("Preprocessing training data....")
    gc.collect()
    data_for_training = orig_data

    target_column = "target"

    if kind == "train":
        data_for_training['target']= np.where((data_for_training['booking_bool'] == 1), 2, np.where((data_for_training['click_bool'] ==1) & (data_for_training['booking_bool'] == 0), 1, 0))

    threshold = 0.9
    #data_for_training = add_date_features(data_for_training)
    data_for_training.drop(labels=["date_time"], axis=1, inplace=True)

    data_for_training = drop_columns_with_missing_data(data_for_training, threshold)

    gc.collect()
    if kind == "train":
        y = data_for_training[target_column].values
    else:
        y = None

    training_set_only_metrics = ["click_bool", "booking_bool", "gross_bookings_usd"]
    columns_to_remove = [
        "date_time",
        "target",
        target_column,
    ] + training_set_only_metrics
    columns_to_remove = [
        c for c in columns_to_remove if c in data_for_training.columns.values
    ]
    data_for_training = data_for_training.drop(labels=columns_to_remove, axis=1)
    return data_for_training, y


def split_train_data(data_for_training, y):

    x1, x2, y1, y2 = train_test_split(data_for_training, y, test_size=0.2, random_state=42)

    srch_id_dest_id_dict = x1.loc[x1["random_bool"] == 0]

    # estimated position calculation
    srch_id_dest_id_dict = x1.loc[x1["random_bool"] == 0]
    srch_id_dest_id_dict = x1.groupby(["srch_destination_id", "prop_id"]).agg(
        {"position": "mean"}
    )
    srch_id_dest_id_dict = srch_id_dest_id_dict.rename(
        index=str, columns={"position": "estimated_position"}
    ).reset_index()
    srch_id_dest_id_dict["srch_destination_id"] = (
        srch_id_dest_id_dict["srch_destination_id"].astype(str).astype(int)
    )
    srch_id_dest_id_dict["prop_id"] = (
        srch_id_dest_id_dict["prop_id"].astype(str).astype(int)
    )
    srch_id_dest_id_dict["estimated_position"] = (
        1 / srch_id_dest_id_dict["estimated_position"]
    )
    x1 = input_estimated_position(x1, srch_id_dest_id_dict)
    x2 = input_estimated_position(x2, srch_id_dest_id_dict)

    groups = x1["srch_id"].value_counts(sort=False).sort_index()
    eval_groups = x2["srch_id"].value_counts(sort=False).sort_index()
    len(eval_groups), len(x2), len(x1), len(groups)

    x1 = remove_columns(x1)
    x2 = remove_columns(x2)
    return (x1, x2, y1, y2, groups, eval_groups, srch_id_dest_id_dict)


def run(output_dir):
    train_csv = pd.read_csv('C:/Users/bruce/Desktop/DMT_Assignment2/random_subset.csv')
    #training_data = load_data(train_csv)
    training_data, y = preprocess_training_data(train_csv)
    method = "dart"
    lr = 0.12
    x1, x2, y1, y2, groups, eval_groups, srch_id_dest_id_dict = split_train_data(
            training_data, y
        )
    model = train_model(
        x1, x2, y1, y2, groups, eval_groups, lr, method, output_dir)
    
    test_csv = pd.read_csv('C:/Users/bruce/Desktop/DMT_Assignment2/random_subset_test.csv')
    #test_data = load_data(test_csv)
    test_data, _ = preprocess_training_data(test_csv, kind="test")
    predict(test_data, srch_id_dest_id_dict, output_dir)
    print("Submit the predictions file submission.csv to kaggle")


run('C:/Users/bruce/Desktop/DMT_Assignment2')
#run('C:/Users/bruce/Desktop/DMT_Assignment2/random_subset.csv', 'C:/Users/bruce/Desktop/DMT_Assignment2/random_subset_test.csv', 'C:/Users/bruce/Desktop/DMT_Assignment2')