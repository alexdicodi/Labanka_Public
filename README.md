# Final Project Interface

https://labanka.herokuapp.com/

# Data analysis
- Document here the project: PD_LOAN_PREDICTION
- Description: Predicting customer default based on Credit Bureau Data and behavioural financial information
- Data Source: Kaggle Competition (https://www.kaggle.com/c/home-credit-default-risk/data)
- Project workflow: Initial Data Exploration > Data Processing > Feature Selection > Model Development > Deployment

# Project Summary

The goal of this project is to create a model capable of predicting if a loan request will end up being a default or not.

Regarding data sources, we can leverage the following information:

- Application form: Main source of customers' personal data (i.e. employment, income amount, sector, family status, etc.)
- Credit Bureau Data: Loan level information of the customers' debts (i.e. Days past due, events timestamp, credit limit, type of credit, etc.)
- Customer behavioural data: Information about previous loans taken by the customer with Home Credit (i.e. pending amount, acquisition channel, instalment plan, etc.)

Concerning the project output, our goal is to have a classification model deployed in the cloud which based on input provided by the customer through a streamlit front-end will authorize or deny the loan.

## Initial Considerations

The initial data exploration of the project focused on answering the following questions:

- Is this a balanced dataset? 
- What role do timestamps deltas play into the credit default prediction?
- What are the default vs. non-default distributions within the customers' personal information categories (i.e. owns house, employment type, etc.)?
- What are the default vs. non-default distributions within the Credit Bureau data (i.e. loan type, limit usage, etc.)?
- What are the default vs. non-default distributions within the Customer behavioural data (i.e. acquisition channel, # of previous loans, etc.)?
- Which features have little or no variance within the dataset?
- What kind of correlations exist within the different data sources?

## Initial Hypothesis

We believe that the most important factors to the default prediction will be:

- DSTI (debt income to service ratio) 
- Employment type
- Employment length
- Previous default history
- Income level
- Other debt products type (i.e. revolving)


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

# Install

Go to `https://github.com/alexdicodi/LABANKA_PUBLIC.git` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:alexdicodi/LABANKA_PUBLIC.git
cd Labanka_Public
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
LABANKA_PUBLIC-run
```
