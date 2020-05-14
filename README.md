# Warsaw Pollution Prediction

**Główne wykorzystane pomysły:**
* **Predykcja tylko jednej kropki na raz.** Poziom smogu w kolejnej godzinie będzie podobny do poziomu z poprzedniej godziny. Z kolei poziom smogu sprzed 24 godzin nie będzie miał większego wpływu na obecny poziom. W związku z tym agregaty muszą mieć jak najmniejszy lag - najlepiej tylko 1 godzinę. Oznacza to, że trzeba przewidywać tylko jedną kropkę na przód, przeliczać agregaty, predykować kolejną kropkę, przeliczać agregaty i tak w kółku aż do spredykowania wszystkich 24 kropek. Zaimplementowałem w ten sposób zarówno finalną predykcję jak i walidację. Planowałem zaimplementować również trening w ten sposób, ale brakło mi czasu.
Feature importances pokazują, że zdecydowanie najważniejszą cechą był właśnie rolling_mean_lag1 - czyli wartość z poprzedniej godziny. Dla pierwszej przewidywanej godziny była to rzeczywista ostatnia wartość, dla wszystkich kolejnych była to wartość przewidziana w poprzednim kroku. 

* **Użycie zewnętrznego zbioru danych z prognozami pogody.** Duży wpływ na poziom smogu ma również pogoda (temperature, prędkość wiatru). Oczywiście nie wiadomo jaka będzie pogoda z wyprzedzeniem. Istnieją jednak prognozy pogody, które są znane z wyprzedzeniem. Nie są one w 100% dokładne, ale powinny być dokładniejsze niż np agregaty z 24 godzinnym lagiem. Użyłem danych z https://www.worldweatheronline.com/developer/api/historical-weather-api.aspx. Muszę przyznać, że nie znalazłem informacji z jak dużym wyprzedzeniem robione były te prognozy, ale zakładam, że z co najmniej 24 godzinnym. Potwierdza to fakt, że każdy wiersz (czyli dane dla każdej godziny) zawierają maksymalną i minimalną temperaturę tego dnia, co sugeruje, że prognoza jest conajmniej na 24 godziny wprzód.

* **Median_by_hour oraz deviation_from_previous.** Median_by_hour jest medianą poziomu pm25 dla konkretnej godziny, wyliczonej osobno dla każdego okna czasowego zakończonego dniem testowym (czyli dla 50 różnych okresów). W związku z tym dane z przyszłości nie są użyte do predykowania przeszłości.  Deviation_from_previous to różnica między medianą poziomu pm25 dla kolejnej następującej godziny i poziomu dla obecnej godziny. Czyli, na przykład, wartość deviation_from_previous dla godziny 14:00 mówi o ile “średnio” poziom smogu był wyższy o 14:00 niż o 13:00.

* **Rolling aggregates z lagiem 24 dla historycznej temperatury (nie prognozowanej).** Nie do końca wiem dlaczego, ale średnia temperatura sprzed 24 godzin miała zauważalny pozytywny wpływ na wynik mimo dużego lagu oraz posiadaniu również prognozowanej temperatury.

\
**Feature importances - gain**
![Gain](https://github.com/MichalPorwisz/warsaw-pollution/blob/master/visualizations/importances_gain_13_22_8.png?raw=true)

\
**SHAP summary plot**\
*https://shap.readthedocs.io/en/latest/*
\
![SHAP summary plot](https://github.com/MichalPorwisz/warsaw-pollution/blob/master/visualizations/importances_shap_13_22_8.png?raw=true)

\
**Wyniki przy usunięciu niektórych cech.**
Z ciekawości sprawdziłem jakie uzyskałbym wyniki, gdybym usunął którąkolwiek z powyżej opisanych grup cech. Wynik przy użyciu wszystkich cech to 11,34 (na private). Wyniki przy usuwaniu cech były następujące (wszystkie na private):
* bez cech rolling_meanX - RMSE powyżej **20**  (co potwierdza wagę tych cech)
* bez uwzględnienia prognozowanej pogody - **15,23**
* bez uwzględnienia median_by_hour oraz deviation_from_previous - **13,64**
* bez rolling agregatów dla historycznej temperatury - **13,88**

Eksperymenty te są o tyle wadliwe, że były wykonane bez hyperparameter searchu (w rozwiązaniu konkursowym sprawdziłem jakieś 10-20 zestawów parametrów - nie jakoś bardzo dużo, ale jednak). Wykorzystane były te same parametry co w ostatecznym rozwiązaniu konkursowym. Mimo wszystko eksperymenty te sugerują jednak, że wszystkie wyżej opisane cechy były istotne dla wyniku. Z drugiej strony mogą też sugerować niestabilność rozwiązania, ale to można by zweryfikować na większym zbiorze danych.

\
**Struktura kodu/najważniejsze skrypty:**
* folder `preprocessing` - zawiera wszystkie skrypty użyte do uzyskania ostatecznej wersji danych
* `predict_one_by_one.py` - skrypt służący do procedury predykowania jednej kropki na przód, przeliczania agregatów z użyciem tej predykcji i dopiero wtedy predykowania kolejnej kropki
* `xgb_custom_training.py` - skrypt z własną implementacją walidacji i early stopping dla trenowania XGB. Potrzebny ze względu na to, że walidacja powinna symulować procedurę finalnej predykcji, tzn. musi wykorzystywać skrypt `predict_one_by_one.py` i podejmować decyzję, kiedy model jest optymalny na podstawie RMSE uzyskanego za pomocą tej procedury.
* `train_xgb.py` - skrypt, w którym użyte są powyższe skrypty w celu dokonania finalnej predykcji

\
**Kroki w celu reprodukowania rozwiązania:**
* ściągnąć kod lokalnie
* uruchomić w terminalu `pip install -r requirements.txt` (w głównym folderze kodu)
* install XGBoost (miałem problemy z instalacją przez pip, więc zainstalowałem przez anacondę - https://anaconda.org/conda-forge/xgboost - ale przez pip też na pewno się da)
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
  * `./preprocessing.sh` (skrypt uruchamia skrypty pythonowe, który użyłem do uzyskania ostatecznej wersji danych)
  * alternatywnie można też odpalić każdy skrypt pythonowy "ręcznie" - w tej samej kolejności co w preprocessing.sh
* Uruchomić skrypt `train_xgb.py`
