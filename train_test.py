import torch
import numpy as np
import tabulate
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
# Обучение модели
def train(
  model,
  train_loader,
  val_loader,
  loss_fn,
  optimizer,
  device,
  epochs
    ):

    for epoch in range(epochs):

            train_labels = np.empty((0, 0))
            train_preds = np.empty((0, 0))
            val_labels = np.empty((0, 0))
            val_preds = np.empty((0, 0))
            total_loss_train, total_loss_val = 0, 0

            print(f'Epoch {epoch + 1}/{epochs}')

            for train_batch in train_loader:

                input_ids = train_batch['input_ids'].to(device)
                mask = train_batch['attention_mask'].to(device)
                targets = train_batch['targets'].to(device)
                outputs = model(input_ids, mask)

                batch_loss = loss_fn(outputs, targets)
                total_loss_train += batch_loss.item()

                train_preds = np.append(train_preds, outputs.argmax(dim=1).detach().cpu().numpy())
                train_labels = np.append(train_labels, targets.cpu().numpy())

                batch_loss.backward()
                optimizer.step()
                model.zero_grad()

            # Валидация модели
            with torch.inference_mode():

                for val_batch in val_loader:

                    input_ids = val_batch['input_ids'].to(device)
                    mask = val_batch['attention_mask'].to(device)
                    targets = val_batch['targets'].to(device)
                    outputs = model(input_ids, mask)

                    batch_loss = loss_fn(outputs, targets)
                    total_loss_val += batch_loss.item()

                    val_preds = np.append(val_preds, outputs.argmax(dim=1).detach().cpu().numpy())
                    val_labels = np.append(val_labels, targets.cpu().numpy())

            # Извлечение необходимых метрик
            train_metrics = get_metrics(train_preds, train_labels)
            val_metrics = get_metrics(val_preds, val_labels)
            # Отображение результатов
            print('\n - Train loss: {:.4f}'.format(total_loss_train / len(train_labels)))
            print('\n - Validation loss: {:.4f}'.format(total_loss_val / len(val_labels)))
            print(tabulate([['Train']+list(train_metrics.values()), ['Val']+list(val_metrics.values())], headers=train_metrics.keys(), floatfmt=".4f"))
# Тестирование модели
def evaluate(
    model,
    test_loader,
    loss_fn,
    device,
    matrix=False
    ):

    test_labels = np.empty((0, 0))
    test_preds = np.empty((0, 0))
    total_loss_test = 0

    model.eval()

    with torch.inference_mode():

        for test_batch in test_loader:

            input_ids = test_batch['input_ids'].to(device)
            mask = test_batch['attention_mask'].to(device)
            targets = test_batch['targets'].to(device)
            outputs = model(input_ids, mask)

            batch_loss = loss_fn(outputs, targets)
            total_loss_test += batch_loss.item()

            test_preds = np.append(test_preds, outputs.argmax(dim=1).detach().cpu().numpy())
            test_labels = np.append(test_labels, targets.cpu().numpy())
    # Излечение необходимых метрик
    test_metrics = get_metrics(test_preds, test_labels)
    # Табулирование метрик
    print(tabulate([list(test_metrics.values())], headers=test_metrics.keys(), floatfmt=".4f"))
    # Построение матрицы неточностей
    if matrix:
        cf_matrix = confusion_matrix(test_labels, test_preds)
        sns.heatmap(cf_matrix, annot=True)

# Получение предсказания для конкретного текста
def get_prediction(tokenized_text, model, device):

    model.eval()

    with torch.inference_mode():

        input_ids = tokenized_text['input_ids'].to(device)
        mask = tokenized_text['attention_mask'].to(device)
        output = model(input_ids, mask)
        pred = output.argmax(dim=1).detach().cpu().numpy()

    return pred[0]

# Вспомогательная функция для расчета метрик
def get_metrics(preds, labels, n_classes=2):

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {'accuracy' : accuracy, 'precision' : precision, 'recall' : recall,
           'f1-score' : f1}
