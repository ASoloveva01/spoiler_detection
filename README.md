# Приложение для обнаружения спойлеров
Данное веб-приложение позволяет проверять текст рецензий на наличие спойлеров на основе предобученной модели rubert-tiny
### Используемые библиотеки:
- beautifulsoup4
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
docker run --name c1 -p 5000:5000 -d spoiler_detection_app
```
## Как пользоваться приложением
Для проверки рецензии на спойлеры введите текст рецензии и нажмите на кнопку "Проверить", далее произойдет перенаправление на страницу с результатом.
![Иллюстрация к проекту](https://github.com/ASoloveva01/spoiler_detection/raw/main/app1.png)   
![Иллюстрация к проекту](https://github.com/ASoloveva01/spoiler_detection/raw/main/app2.png)
## Датасет
Данные для обучения были спарсены с сайта: https://www.livelib.ru/reviews.  
Датасет содержит следующие поля:
- **review_text:** Содержит текст рецезии.
- **is_spoiler:** Имеет два значения для обозначения наличия спойлера(1) и его отсутсвия(0).  
Ниже представлены графики, описывающие данные.  
![Иллюстрация к проекту](https://github.com/ASoloveva01/spoiler_detection/raw/main/classes_frequency.png)  
![Иллюстрация к проекту](https://github.com/ASoloveva01/spoiler_detection/raw/main/words_per_review.png)   
## Результаты
<table>
        <thead>
            <tr>
                <th scope="col"></th>
                <th scope="col">Accuracy</th>
                <th scope="col">Precision</th>
                <th scope="col">Recall</th>
                <th scope="col">F1-Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope="row">Train</th>
                <td>0,8556</td>
                <td>0,8664</td>
                <td>0,8341</td>
                <td>0,8495</td>
            </tr>
            <tr>
                <th scope="row">Val</th>
                <td>0,6</td>
                <td>0,6</td>
                <td>0,4377</td>
                <td>0,4867</td>
            </tr>
            <tr>
                <th scope="row">Test</th>
                <td>0,5306</td>
                <td>0,5306</td>
                <td>0,12</td>
                <td>0,2069</td>
            </tr>
        </tbody>
</table>
