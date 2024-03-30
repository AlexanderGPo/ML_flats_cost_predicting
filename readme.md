# ML. Предсказание рыночной цены на квартиру

## Описание программы
Программа представляет из себя модель (построенную на `RandomForestRegressor`)

## Взаимодействие с программой
В файле `main.py` необходимо в `pd.Series` `info_for_predicting` указать необходимые данные, после чего запустить программу. Данные должны строго соответствовать следующему описанию:
   - `author_type` - "тип" человека, сдающего недвижимость (возможные варианты: 'homeowner', 'official_representative', 'real_estate_agent', 'realtor', 'developer')
   - `floor` - этаж представленной квартиры
   - `floors_count` - количество этажей в здании
   - `rooms_count` - количество комнат в квартире
   - `total_meters` - площадь квартиры в квадратных метрах
   - `price` - цена квартиры. таргет
   - `address` - адрес в формате `г. Москва, ул. ______, дом __`

Обращаем внимание на то, что программа работает пока только с городом **Москва**. Соответственно, все адреса должны быть московскими.
## Содержание репозитория
Папка `data` - информация, необходимая для обучения модели. Содержит таблицу `data.csv` представляет из себя набор данных размером `14 x 26970`.
    <br><br>
    Таблица имеет следующие столбцы:
   - `author_type` - "тип" человека, сдающего недвижимость (возможные варианты: 'homeowner', 'official_representative', 'real_estate_agent', 'realtor', 'developer')
   - `floor` - этаж представленной квартиры
   - `floors_count` - количество этажей в здании
   - `rooms_count` - количество комнат в квартире
   - `total_meters` - площадь квартиры в квадратных метрах
   - `price` - цена квартиры. таргет
   - `lat` - широта квартиры
   - `lon` - долгота квартиры 
## Парсинг сайта
Информация, представленная в файле `data.csv` была получена с сайта [Циан](cian.ru), с помощью бибилотеки `cianparser`. Информация собиралась с каждой станции метро. Ниже представлен использовавшийся для этого код.
```
import cianparser

stations = cianparser.list_metro_stations()['Московский']
moscow_parser = cianparser.CianParser(location="Москва")

for current_station in stations:
    data = moscow_parser.get_flats(deal_type="sale", rooms = 'all', with_saving_csv=True, additional_settings={"metro": "Московский", "metro_station": current_station[0]}
```
Важно отметить, что код несколько раз "падал" из-за появления Captcha на сайте, поэтому с "первого раза" код может не сработать.