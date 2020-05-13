# warsaw-pollution

* `pip install -r requirements.txt`
* install XGBoost (miałem problemy z instalacją przez pip, więc zainstalowałem przez anacondę - https://anaconda.org/conda-forge/xgboost - ale przez pip też na pewno się da
* wrzucić train_warsaw.h5 i test_warsaw.h5 do katalogu input (nie wiem czy można udostępniać te dane publicznie, więc nie udostępniam)
* do katalogu data_processed/forecasts wrzucić plik o nazwie warsaw_full.csv (nie wiem czy można udostępniać te dane publicznie, więc nie udostępniam). Instrukcja do pozyskania danych poprzez API:
  * Zarejestrować się w serwisie Historical Weather Forecast - https://www.worldweatheronline.com/developer/api/historical-weather-api.aspx
  * W zmiennej `api_key` w skrypcie `preprocessing/retrieve_forecast_data.py`, ustawić api key pozyskany z tego serwisu
  * Uruchomić skrypt `preprocessing/retrieve_forecast_data.py`
  * Z nowszymi wersjami pythona wyskakuje błąd: `AttributeError: module 'urllib' has no attribute 'request'`. Rozwiązuje się 
  go idąc do kodu wewnątrz modułu www_hist, plik `_init_.py` (z którego pochodzi błąd) i 
  zamieniając `import urllib` na
`import urllib.request`
* Uruchomić preprocessing:
  * `cd preprocessing` 
  * `chmod +x preprocessing.sh` 
  * `./preprocessing.sh`
* Uruchomić skrypt `train_xgb.py`
