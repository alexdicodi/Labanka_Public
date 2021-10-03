FROM python:3.8.6-buster

COPY API /API
COPY Train/no_bureau_model_lgbm.sav /no_bureau_model_lgbm.sav
COPY requirements.txt /requirements.txt
COPY LABANKA_PUBLIC /LABANKA_PUBLIC

RUN pip install -r requirements.txt

CMD uvicorn API.fast_api_labanka:app --host 0.0.0.0 --port $PORT