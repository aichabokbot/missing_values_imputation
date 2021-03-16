# Natixis Data Challenge - Risk

## Installation

1) Clone the repository on your local machine
2) Cd in the project folder
3) Download [blob_credentials.py](https://gitlab.repositories.hec.hfactory.io/data-challenges/natixis-risks-2021/group1/credentials/-/raw/master/blob_credentials.py) and make it executable 
4) Run `pip install -r requirements.txt`
5) Run `notebook/intro.ipynb` to download the data
6) Run `nbstripout --install` to avoid conflicts with .ipynb files

## Project Description

The goal of this project is to establish an optimal, robust method to impute missing data in Financial Time-Series.  

This would allow Natixis to increase the accuracy and robustness of Risk Management Models, but also Increase the performance of predictive models, enterprise-wide.

## Dataset Description

The data consists of 1504 Time-Series with daily granularity across 6 different asset classes:
- Bonds
- CDS Spread
- Commodities
- FX Rates
- Stocks
- Yield Curves