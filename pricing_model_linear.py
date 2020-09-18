import pickle

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight, resample, shuffle


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModelLinear():
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # to store the scalar object
        self.scalar = None
        # to store the result columns after encoding training, to make sure both training and testing
        # are having exact structure
        self.train_columns = None 
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = None # ADD YOUR BASE CLASSIFIER HERE

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw, train=False):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # drop unwanted columns
        for col in columns_to_drop:
            if col in X_raw.columns:
                X_raw.drop(columns=[col], inplace=True)
        
        
        #fill missing values with median (to avoid being biased towards outliers)
        for column in X_raw.columns:
            if is_numeric_dtype(X_raw[column]):
                X_raw[column] = X_raw[column].fillna(X_raw[column].median())
        
        #encode categorical variables and scale all columns
        columnsToEncode = list(X_raw.select_dtypes(include=['category','object']))
        tmp_dummies = pd.get_dummies(X_raw[columnsToEncode], drop_first=True)
        tmp_dummies.head()
        X_raw = pd.concat([X_raw, tmp_dummies], axis=1)
        X_raw = X_raw.drop(columnsToEncode, axis=1)
        if train:
            self.train_columns = X_raw.columns
            scalar = StandardScaler()
            X_raw.loc[:,:] = scalar.fit_transform(X_raw)
            self.scalar = scalar
        else:
            # if there's missing columns in the testing after encoding, to ensure both are having the same structure
            # Get missing columns in the training test
            missing_cols = set(self.train_columns) - set(X_raw.columns)
            # Add a missing column in test set with default value equal to 0
            for c in missing_cols:
                X_raw[c] = 0
            # Ensure the order of column in the test set is in the same order than in train set
            X_raw = X_raw[self.train_columns]
            X_raw.loc[:,:] = self.scalar.transform(X_raw)
        return X_raw 

    def fit(self, X, y, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.median(claims_raw[nnz])*.4


        ###### Handle train
        X_train_clean = self._preprocessor(X, train=True)
        # concatenate our training data back together
        X = pd.concat([X_train_clean, y], axis=1)
        # separate minority and majority classes
        class_0 = X[X.made_claim==0]
        class_1 = X[X.made_claim==1]
        # upsample minority
        df_train = resample(class_1,
                                replace=True, # sample with replacement
                                n_samples=len(class_0), # match number in majority class
                                random_state=1) # reproducible results
        # combine majority and upsampled minority
        df_train = pd.concat([df_train, class_0])
        df_train = shuffle(df_train)

        X_train = df_train.drop(columns=['made_claim'])
        y_train = df_train['made_claim'].copy()

        model = LogisticRegressionCV(cv=10, solver='lbfgs')

        self.base_classifier = model

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_train, y_train)
        else:
            self.base_classifier = self.base_classifier.fit(X_train, y_train)

        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================

        X_clean = self._preprocessor(X_raw)

        return self.base_classifier.predict_proba(X_clean)[:,1]# return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        return (self.predict_claim_probability(X_raw) * self.y_mean).flatten()

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)

def load_model_linear():
    return pickle.load(open("part3_pricing_model_linear.pickle", "rb"))


columns_to_drop = ['pol_payd','pol_pay_freq','id_policy', 
    'pol_insee_code','vh_make','vh_model',
    'regional_department_code','drv_sex2',
    'commune_code','canton_code','city_district_code']

if __name__ == '__main__':
    df = pd.read_csv('sample_claim_data_v2.csv')
    df.drop(columns=columns_to_drop, inplace=True)

    X = df.drop(columns=['made_claim','claim_amount'])
    y = df['made_claim'].copy()
    claims_raw = df['claim_amount'].copy()

    p = PricingModelLinear()
    p.fit(X, y, claims_raw)
    p.save_model()
    #result = p.predict_premium(X_test)
    #print(result.shape)
