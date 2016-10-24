import pandas as pd
from os import path as osp

BASE_BATH = osp.dirname(osp.realpath(__file__))
DATA_PATH = BASE_BATH + '/../data/movie_metadata.csv'
FINAL_DATA_PATH = BASE_BATH + '/../data/processed_movie_metadata.csv'

# These features are detailed in the IMDB_5000_EDA notebook
# (how they are obtained, what they mean...)

useless_features = ["aspect_ratio", "movie_imdb_link",
                    "facenumber_in_poster"]
string_features = ["movie_title", "plot_keywords"]
highly_correlated_features = ["cast_total_facebook_likes"]

removed_features = (["gross", "budget", "actor_1_name",
                     "actor_2_name",
                     "actor_3_name", "director_name"] + useless_features +
                    highly_correlated_features + string_features)

filtered_categorical_features = ["color", "genres", "language", "country",
                                 "content_rating"]


def make_dummies(input_df):
    df = input_df.copy()
    dummies = []
    for col in filtered_categorical_features:
        if col != 'genres':
            dummies.append(pd.get_dummies(df[col]))
        else:
            dummies.append(df.genres.str.get_dummies(sep='|'))
    return pd.concat(dummies, axis=1)

# -----------------------------------------------#


def process_data():

    # Read the dataset, remove columns and drop NaNs

    cleaned_df = (pd.read_csv(DATA_PATH, encoding='UTF-8')
                    .drop(removed_features, axis=1)
                    .dropna())

    # Construct dummies for the categorical features
    dummies_df = cleaned_df.pipe(make_dummies)

    # Add the dummies features to the cleaned dataframe and save it
    (pd.concat([cleaned_df.drop(filtered_categorical_features, axis=1),
                dummies_df], axis=1)
        .to_csv(FINAL_DATA_PATH, index=False))


if __name__ == "__main__":
    process_data()
