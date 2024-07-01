# import project libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import re

from .constants import RELEVANT_PREPROCESS_COLUMNS


# Apply msrp value
def map_msrp(msrp):
    """
    Replace 0 values by null

    :param msrp: manufacturer's suggested retail price
    """
    if msrp == 0:
        return np.nan
    return msrp


def clean_exterior_color(exterior_color):
    # Check if value is empty
    if pd.isna(exterior_color):
        return 'unknown'
    # Convert interior_color to lower case
    exterior_color = exterior_color.lower()
    # Remove special characters
    exterior_color = re.sub(r'[\W_+w/\/]', ' ', exterior_color)
    # Remove double spaces
    exterior_color = re.sub(r'\s+', ' ', exterior_color)
    # Apply trim
    exterior_color = exterior_color.strip()
    # Return formated text
    return exterior_color


def get_exterior_color_phrase_vector(exterior_color_phrase, model):
    exterior_color_words = exterior_color_phrase.split()
    exterior_color_word_vectors = [model.wv[word] for word in exterior_color_words if word in model.wv]
    if not exterior_color_word_vectors:
        print(f"No words found in model for phrase: {exterior_color_phrase}")
        return np.nan
    return sum(exterior_color_word_vectors) / len(exterior_color_word_vectors)


def clean_interior_color(interior_color):
    # Check if value is empty
    if pd.isna(interior_color):
        return 'unknown'
    # Convert interior_color to lower case
    interior_color = interior_color.lower()
    # Remove special characters
    interior_color = re.sub(r'[\W_+w/\/]', ' ', interior_color)
    # Remove double spaces
    interior_color = re.sub(r'\s+', ' ', interior_color)
    # Return formated text
    return interior_color


def get_interior_color_phrase_vector(interior_color_phrase, model):
    interior_color_words = interior_color_phrase.split()
    interior_color_word_vectors = [model.wv[word] for word in interior_color_words if word in model.wv]
    if not interior_color_word_vectors:
        print(f"No words found in model for phrase: {interior_color_phrase}")
        return np.nan
    return sum(interior_color_word_vectors) / len(interior_color_word_vectors)


def map_drivetrain(drivetrain):
    """
    Group the drive trian by categories

    :param drivetrain: Car drive train

    :return: Grouped drive train
    """
    if pd.isna(drivetrain):
        return np.nan
    # Apply lower case and replace special characters
    drivetrain = str(drivetrain).lower().replace('-', ' ')

    match drivetrain:
        case 'all wheel drive' | 'four wheel drive' | 'awd' | '4wd' | '4x2' | 'all wheel drive with locking and limited slip differential' | '4matic':
            return 'All-wheel Drive'
        case 'rear wheel drive' | 'rwd':
            return 'Rear-wheel Drive'
        case 'front wheel drive' | 'fwd' | 'front wheel drive':
            return 'Front-wheel Drive'
        case 'unknown':
            return np.nan
        case _:
            raise Exception(f"No expected drive train: {drivetrain}")


def clean_cat(cat):
    # Check if value is empty
    if pd.isna(cat):
        return 'unknown'
    # Convert cat to lower case
    cat = cat.lower()
    # Split by '_' and join again by ' '
    cat = ' '.join(cat.split('_'))
    # Remove double spaces
    cat = re.sub(r'\s+', ' ', cat)
    # Return formated text
    return cat


# Calculate the vectors feature avegare
def get_cat_phrase_vector(cat_phrase, model):
    cat_words = cat_phrase.split()
    cat_word_vectors = [model.wv[word] for word in cat_words if word in model.wv]
    if not cat_word_vectors:
        print(f"No words found in model for phrase: {cat_phrase}")
        return np.nan
    return sum(cat_word_vectors) / len(cat_word_vectors)


def map_fuel_type(fuel_type):
    """
    Group by fuel types

    :param fuel_type: Car fuel type

    :return Fuel type category
    """
    if pd.isna(fuel_type):
        return np.nan

    match fuel_type:
        case 'Gasoline' | 'Gasoline Fuel' | 'Diesel' | 'Premium Unleaded' | 'Regular Unleaded' | 'Premium Unleaded' | 'Diesel Fuel':
            return 'Gasoline'
        case 'Electric' | 'Electric with Ga':
            return 'Electric'
        case 'Hybrid' | 'Plug-In Hybrid' | 'Plug-in Gas/Elec' | 'Gas/Electric Hyb' | 'Hybrid Fuel' | 'Bio Diesel' | 'Gasoline/Mild Electric Hybrid' | 'Natural Gas':
            return 'Hybrid'
        case 'Flexible Fuel' | 'E85 Flex Fuel' | 'Flexible':
            return 'Flexible'
        case _:
            print(f"No expected fuel type: {fuel_type}")
            return np.nan


def map_stock_type(stock_type):
    """
    Map stock_type

    :param stock_type: stock type New/Used

    :return Binary stock_type
    """
    if pd.isna(stock_type):
        return np.nan

    match stock_type:
        case 'New':
            return True
        case 'Used':
            return False
        case _:
            raise Exception(f"No expected stock type: {stock_type}")


def preprocess(cars_df, make_valid_categories, hasher_model_model: FeatureHasher, exterior_color_vector_size: int,
               w2v_exterior_color: Word2Vec, interior_color_vector_size: int, w2v_interior_color: Word2Vec,
               make_encoder: OneHotEncoder, drivetrain_encoder: OneHotEncoder, bodystyle_encoder: OneHotEncoder,
               cat_vector_size: int, w2v_cat: Word2Vec, fuel_type_encoder: OneHotEncoder, imputer: IterativeImputer,
               iso_forest: IsolationForest, scaler: MinMaxScaler, scale_data=False):
    """
    Pre process cars data

    :param cars_df: Cars input data
    :param make_valid_categories: Make valid categories
    :param hasher_model_model: Hasher model
    :param exterior_color_vector_size: Word2Vec exterior color vector size
    :param w2v_exterior_color: Word2Vec exterior color model
    :param interior_color_vector_size: Word2Vec interior color vector size
    :param w2v_interior_color: Word2Vec interior color model
    :param make_encoder: Make encoder model
    :param drivetrain_encoder: Drivetrain encoder model
    :param bodystyle_encoder: Body style encoder model
    :param cat_vector_size: Word2Vec cat vector size
    :param w2v_cat: Word2Vec category
    :param fuel_type_encoder: Fuel type encoder model
    :param imputer: Imputation model
    :param iso_forest: Outlier detection model
    :param scaler: Scaler model
    :param scale_data: Scale data using min max scaler

    :return: Cars processed data

    """
    print("Star Inference preprocess")

    print("####### Validate data")
    print("Validate dataset before preprocessing")
    # Check if dataframe has the columns needed
    assert set(cars_df.columns) == set(RELEVANT_PREPROCESS_COLUMNS), 'Input has invalid columns'
    # Validate features
    assert cars_df.loc[cars_df['drivetrain'].isna()].shape[0] == 0, "No empty drive train"
    assert cars_df.loc[cars_df['fuel_type'].isna()].shape[0] == 0, "No empty fuel type"
    # Check make_valid_categories is not empty then filter validate makes
    if make_valid_categories:
        if not cars_df['make'].isin(make_valid_categories).all():
            raise Exception(f"No valid make: {make_valid_categories}")

    print("####### Transform data")
    # ### Apply Features transformation
    # Apply msrp transformation
    print("Apply msrp transformation")
    cars_df['msrp'] = cars_df['msrp'].map(map_msrp)
    # Apply model transformation
    print("Apply model transformation")
    cars_model_hashed = hasher_model_model.transform(cars_df['model'].apply(lambda x: {x: 1}).tolist())
    # Generate model hashed dataframe
    cars_model_hashed_df = pd.DataFrame(cars_model_hashed.toarray(),
                                        columns=[f'model_hashed_{i}' for i in range(cars_model_hashed.shape[1])],
                                        index=cars_df.index)
    cars_df = pd.concat([cars_df, cars_model_hashed_df], axis=1)

    # Apply exterior_color transformation
    print("Apply exterior_color transformation")
    # Apply lower case and remove special characters
    cars_df['exterior_color'] = cars_df['exterior_color'].apply(clean_exterior_color)
    # Calculate the vector for each interior color
    cars_exterior_color_vectors_s = cars_df['exterior_color'].apply(
        lambda ic: get_exterior_color_phrase_vector(ic, w2v_exterior_color))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * exterior_color_vector_size
    cars_exterior_color_vectors_s = cars_exterior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    cars_exterior_color_df = pd.DataFrame(cars_exterior_color_vectors_s.values.tolist(),
                                          columns=[f'exterior_color_x{i}' for i in
                                                   range(len(cars_exterior_color_vectors_s.iloc[0]))],
                                          index=cars_df.index)
    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, cars_exterior_color_df], axis=1)

    # Apply interior_color transformation
    print("Apply interior_color transformation")
    # Apply lower case and remove special characters
    cars_df['interior_color'] = cars_df['interior_color'].apply(clean_interior_color)
    # Calculate the vector for each interior color
    cars_interior_color_vectors_s = cars_df['interior_color'].apply(
        lambda ic: get_interior_color_phrase_vector(ic, w2v_interior_color))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * interior_color_vector_size
    cars_interior_color_vectors_s = cars_interior_color_vectors_s.apply(
        lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    cars_interior_color_df = pd.DataFrame(cars_interior_color_vectors_s.values.tolist(),
                                          columns=[f'interior_color_x{i}' for i in
                                                   range(len(cars_interior_color_vectors_s.iloc[0]))],
                                          index=cars_df.index)
    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, cars_interior_color_df], axis=1)

    # Apply drive train transformation
    print("Apply drivetrain transformation")
    cars_df['drivetrain'] = cars_df['drivetrain'].map(map_drivetrain)
    # Transform the data
    cars_drivetrain_encoded_data = drivetrain_encoder.transform(cars_df[['drivetrain']])
    # Convert the drivetrain encoded data into a DataFrame
    cars_drivetrain_encoded_df = pd.DataFrame(cars_drivetrain_encoded_data,
                                              columns=drivetrain_encoder.get_feature_names_out(['drivetrain']),
                                              index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, cars_drivetrain_encoded_df], axis=1)

    # Apply make transformation
    print("Apply make transformation")
    # Fit and transform the data
    cars_make_encoded_data = make_encoder.transform(cars_df[['make']])
    # Convert the drivetrain encoded data into a DataFrame
    cars_make_encoded_df = pd.DataFrame(cars_make_encoded_data, columns=make_encoder.get_feature_names_out(['make']),
                                        index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, cars_make_encoded_df], axis=1)

    # Apply bodystyle transformation
    print("Apply bodystyle transformation")
    # Fit and transform the data
    cars_bodystyle_encoded_data = bodystyle_encoder.fit_transform(cars_df[['bodystyle']])
    # Convert the drivetrain encoded data into a DataFrame
    cars_bodystyle_encoded_df = pd.DataFrame(cars_bodystyle_encoded_data,
                                             columns=bodystyle_encoder.get_feature_names_out(['bodystyle']),
                                             index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, cars_bodystyle_encoded_df], axis=1)

    # Apply cat transformation
    print("Apply cat transformation")
    # Apply lower case and remove special characters
    cars_df['cat'] = cars_df['cat'].apply(clean_cat)
    # Calculate the vertor for each cat
    cars_cat_vectors_s = cars_df['cat'].apply(lambda ic: get_cat_phrase_vector(ic, w2v_cat))
    # Replace the nan values with an array of (0,0,0)
    base_invalid_value = [0] * cat_vector_size
    cars_cat_vectors_s = cars_cat_vectors_s.apply(lambda x: x if isinstance(x, np.ndarray) else base_invalid_value)
    # Generate the interior color df using the transformed feature vectors
    cars_cat_data = pd.DataFrame(cars_cat_vectors_s.values.tolist(),
                                 columns=[f'cat_x{i}' for i in range(len(cars_cat_vectors_s.iloc[0]))],
                                 index=cars_df.index)
    # Concatenate the dataframes
    cars_df = pd.concat([cars_df, cars_cat_data], axis=1)

    # Apply fuel type transformation
    print("Apply fuel_type transformation")
    cars_df['fuel_type'] = cars_df['fuel_type'].map(map_fuel_type)

    # Encode OneHotEncoder drivetrain
    cars_fuel_type_encoded_data = fuel_type_encoder.transform(cars_df[['fuel_type']])
    # Convert the drivetrain encoded data into a DataFrame
    cars_fuel_type_encoded_df = pd.DataFrame(cars_fuel_type_encoded_data,
                                             columns=fuel_type_encoder.get_feature_names_out(['fuel_type']),
                                             index=cars_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    cars_df = pd.concat([cars_df, cars_fuel_type_encoded_df], axis=1)

    # Apply binary transformation
    print("Apply stock_type transformation")
    cars_df['stock_type'] = cars_df['stock_type'].map(map_stock_type)

    # Remove transformed columns
    cars_df.drop(
        columns=['model', 'exterior_color', 'interior_color', 'drivetrain', 'make', 'bodystyle', 'cat', 'fuel_type'],
        inplace=True)

    print("####### Imputate missing data")
    print("Apply Iterative imputation")
    # Apply imputation
    cars_df_trans = imputer.transform(cars_df)
    # transform the dataset
    cars_df = pd.DataFrame(cars_df_trans, columns=cars_df.columns, index=cars_df.index)

    print("####### Detect outliers data")
    # ### Outliers Detection
    print("Apply Outlier Detection")
    # Remove outliers
    cars_outliers_s = iso_forest.predict(cars_df)
    if cars_outliers_s.shape[0] > 0:
        print(f"There are {cars_outliers_s.shape[0]} outliers instances as input, the accuracy could decrease")

    # Scale data if needed
    if scale_data:
        print("####### Scale data")
        print("Apply Scale Min/Max Transformation")
        # Apply scale transformation
        cars_df_trans = scaler.transform(cars_df)
        # transform the dataset
        cars_df = pd.DataFrame(cars_df_trans, columns=cars_df.columns, index=cars_df.index)

    print("Preprocess Inference completed")

    # Return preprocess data
    return cars_df
