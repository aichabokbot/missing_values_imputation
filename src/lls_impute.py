import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


class lls_imputation:

    def __init__(self):
        pass

    def na_correlation(self, security_type, x, y):
        """
        Use multiple imputation to estimate the correlation coefficient columns x and y.
        y is the column we want to impute the missing values for.
        x is a column for which we want to know the correlation with y.
        """

        ## Columns of interest
        first_valid_index = security_type[y].first_valid_index()
        df = (security_type.apply(lambda x: x.first_valid_index()) <= first_valid_index).to_frame()
        cols_to_keep = df[df[0] == True].index

        if x == y:
            return np.nan

        elif str(x) not in cols_to_keep:
            return np.nan
        else:

            Z = security_type.loc[first_valid_index:, cols_to_keep]
            X = Z.loc[:, (str(x), str(y))].values

            ## The indices of cases with no missing values in columns 1 and 2
            ii = np.flatnonzero(np.isfinite(Z.loc[:, (str(x), str(y))]).all(1))

            ## The correlation coefficients for complete case analysis
            r = np.corrcoef(X[ii, 0], X[ii, 1])[0, 1]

            return r

    def k_LS_neighbors(self, col, k, correlations):
        return correlations[col].sort_values(ascending=False)[:k].index.values

    def impute_na(self, security_type, col, k, correlations):
        Z = security_type.loc[security_type[col].first_valid_index():, :]
        Z[Z.columns.difference([col])] = Z[Z.columns.difference([col])].interpolate()

        na_indexes = np.append(Z[col][Z[col].isna()].index.values, Z.shape[0])

        for i in range(len(na_indexes) - 1):
            w = Z.loc[na_indexes[i] + 1:na_indexes[i + 1] - 1, col]
            A = Z.loc[na_indexes[i] + 1:na_indexes[i + 1] - 1,
                self.k_LS_neighbors(col, k, correlations).astype('str')].T
            b = Z.loc[na_indexes[i], self.k_LS_neighbors(col, k, correlations)]
            missing_value = b.T @ np.linalg.pinv(A.T) @ w
            Z.loc[na_indexes[i], col] = missing_value

        return pd.concat([security_type.loc[:security_type[col].first_valid_index() - 1, col],
                          Z[col]], axis=0)

    def X_impute(self, security_type, k):
        correlations = pd.DataFrame(index=security_type.columns, columns=security_type.columns)
        for i in correlations.index:
            for j in correlations.columns:
                correlations.loc[i, j] = self.na_correlation(security_type, i, j)

        X_imputed = pd.DataFrame(index=security_type.index, columns=security_type.columns)
        for col in X_imputed.columns:
            X_imputed[col] = self.impute_na(security_type, col, k, correlations)

        return X_imputed


if __name__ == '__main__':
    k = 3
    dataset_challenge = pd.read_csv("../data/raw/Risques_2/data_set_challenge.csv")
    mapping = pd.read_csv("../data/raw/Risques_2/final_mapping_candidat.csv")

    result = dataset_challenge.iloc[:, 0]
    for Type in mapping.Type.unique():
        security_type = dataset_challenge[mapping[mapping.Type == Type].mapping_id.values.astype("str")]
        result = pd.concat([result, lls_imputation().X_impute(security_type, k)], axis=1)
        print(Type, ': done')

    result.to_csv("../data/result_lss_imputation.csv", index=False)
