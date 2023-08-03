# Приложение для обнаружения спойлеров
Данное веб-приложение позволяет проверять текст рецензий на наличие спойлеров на основе предобученной модели rubert-tiny
### Используемые библиотеки:
- beautifulsoup
- flask
- numpy
- pandas
- requests
- scikit_learn
- seaborn
- tabulate
- torch
- transformers
## Как установить и запустить
```python
# Клонирование репозитория
git clone https://github.com/ASoloveva01/spoiler_detection
cd spoiler_detection

# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения
python app.py
```
## Использование Docker
```python
# Создание образа
docker build -t spoiler_detection_app .

# Запуск контейнера
docker run --p 8000:8080 --d --name с1 spoiler_detection_app
```
## Как пользоваться приложением
Для проверки рецензии на спойлеры введите текст рецензии и нажмите на кнопку "Проверить".
![Иллюстрация к проекту](https://github.com/ASoloveva01/spoiler_detection/raw/main/app1.png)   
![Иллюстрация к проекту]((https://github.com/ASoloveva01/spoiler_detection/raw/main/app2.png)
## Датасет
Данные для обучения были спарсены с сайта: https://www.livelib.ru/reviews.  
Датасет содержит следующие поля:
- **review_text:** Содержит текст рецезии.
- **is_spoiler:** Имеет два значения для обозначения наличия спойлера(1) и его отсутсвия(0).

