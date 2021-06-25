import os
import math
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from transformers import TrainingArguments, Trainer, BertTokenizer, BertForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 200


def load_dataframe(config, test = False):
    if test:
        path = os.getcwd() + '/data/no_test.csv' # Always testing on Norwegian dataset
    else:
        path = os.getcwd() + config["data_dir"] + '.csv'

    X = config["text_column"]
    y = config["target_column"]

    df = pd.read_csv(path)[[X, y]]
    #df = df.sample(frac = 0.03, random_state = seed) #added to test locally w/o all data

    if y == "hateful":
        df = df[df.hateful != -1]
    df.columns = ["text", "label"]

    return df


def configure_training_set(train, config):
    '''
    Configures training data frame based on config file
    Adds Danish data to the training dataset if translator in ["translatepy", "opus-mt", "418M", "1.2B", "cleaned"]
    Trains on only Danish data if train_dk in config
    Undersamples to 2/3 majority class and the rest predictive class if undersample in config
    '''

    if config["translator"] in ["translatepy", "opus-mt", "418M", "1.2B", "cleaned"]:
        dk = pd.read_csv(os.getcwd() + '/data/dk.csv')[[config["translator"], config['target_column']]]

        if config["target_column"] == "hateful":
            dk = dk[dk.hateful != -1]

        dk.columns = ["text", "label"]
        train = pd.concat([train, dk]).sample(frac=1, random_state=seed)

        if "train_dk" in config:
            train = dk

    if 'undersample' in config:
        n_positives = len(train[train["label"] == 1])
        positives = train.iloc[(train["label"] == 1).values]
        negatives = train.iloc[(train["label"] == 0).values].sample(n=n_positives*2)
        train = pd.concat([positives, negatives]).sample(frac=1)

    return train


def encode_datasets(train_data, dev_data, test_data, config, summary = True):
    '''
    THIS FUNCTION IS ADOPTED W/ PERMISSION FROM KUMMERVOLD ET AL 2021 (https://github.com/NBAiLab/notram)
    Encodes data into tokens and creates pytorch datasets
    train_dataset: used to train the model
    dev_dataset: data on which to evaluate the loss and any model metrics at the end of each epoch.
    test_data: data to evaluate the model
    '''

    tokenizer = BertTokenizer.from_pretrained(config["model_name"])

    # Preprocess data
    X_train = list(train_data["text"])
    y_train = list(train_data["label"])

    X_val = list(dev_data["text"])
    y_val = list(dev_data["label"])
    X_test = list(test_data["text"])

    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=config["max_seq_length"])
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=config["max_seq_length"])
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=config["max_seq_length"])

    # Create torch dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels:
                item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.encodings["input_ids"])

    train_dataset = Dataset(X_train_tokenized, y_train)
    dev_dataset = Dataset(X_val_tokenized, y_val)
    test_dataset = Dataset(X_test_tokenized)
    if summary:
        print(
            f'The dataset is imported.\n\nThe training dataset has {len(train_dataset)} items.\nThe development dataset has {len(dev_dataset)} items. \nThe test dataset has {len(test_dataset)} items')
        steps = math.ceil(len(train_dataset) / config["batch_size"])
        num_warmup_steps = int(0.1 * (steps * config["num_epochs"]))
        print(
            f'You are planning to train for a total of {steps} steps * {config["num_epochs"]} epochs = {config["num_epochs"] * steps} steps. Warmup is {num_warmup_steps}, {math.ceil(100 * num_warmup_steps / (steps * config["num_epochs"]))}%. We recommend at least 10%.')
        print("\n")

    return train_dataset, dev_dataset, test_dataset


def compute_metrics(p):
    '''
    THIS FUNCTION IS ADOPTED W/ PERMISSION FROM KUMMERVOLD ET AL 2021 (https://github.com/NBAiLab/notram)
    Calculates accuracy, precision, and recall
    '''
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def save_model(model, config):
    '''
    Saves fine-tuned model to file specified in model_folder in config
    '''
    print("\n...............................\n")
    print("Saved model to " + config["model_folder"])
    print("\n...............................\n")

    try:
        torch.save(model.model, os.getcwd() + "/" + config["model_folder"] + "model.pth")
    except:
        print("Could not save model")


def load_model(config):
    '''
    Loads fine-tuned model from file specified in model_folder in config
    '''
    print("\n...............................\n")
    print("Loading model from " + config["model_folder"])
    print("\n...............................\n")
    model = torch.load(config["model_folder"]+ "model.pth")

    return model


def fine_tune(config, train_dataset, dev_dataset, save=True):
    '''
    THIS FUNCTION IS PARTLY ADOPTED W/ PERMISSION FROM KUMMERVOLD ET AL 2021 (https://github.com/NBAiLab/notram)
    Fine-tunes model defined in config file on the train_dataset
    '''

    steps = round(len(train_dataset) / int(config["batch_size"]))
    num_warmup_steps = round(0.1 * (steps * int(config["num_epochs"])))# 10%

    model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)

    # Define Trainer
    args = TrainingArguments(
        report_to="none",
        output_dir="output",
        logging_strategy= "no",
        evaluation_strategy="no",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["init_lr"],  # The default here is linear decay to 0.
        warmup_steps=num_warmup_steps,
        num_train_epochs=config["num_epochs"],
        save_steps=steps,  # Only saves at the end
        seed=0,
        disable_tqdm=True, # Changed to remove statistic reportings to wandb
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # Train pre-trained model
    trainer.train()

    print(f'The training has finished training after {config["num_epochs"]} epochs.')
    if save:
        save_model(trainer, config)

    return trainer


def save_predictions(pred, config):
    '''
    Saves predictions to file pred.csv in model_folder specified in config
    '''
    print("\n...............................\n")
    print("Writing predictions to file pred.csv in " + config["model_folder"])
    print("\n...............................\n")
    try:
        np.savetxt(os.getcwd() + "/" + config["model_folder"] + "pred.csv", pred, delimiter=",")
    except:
        print("Could not save predictions....")
    print("..")


def classify(config, trainer, test_dataset, test_data, print_eval=False, save = True):
    '''
    THIS FUNCTION IS PARTLY ADOPTED W/ PERMISSION FROM KUMMERVOLD ET AL 2021 (https://github.com/NBAiLab/notram)
    Predicts test dataset using the fine-tuned trainer
    config:
    trainer: used to predict class of test data
    test_encodings: encoded text from test dataset used to predict
    test_data: only needed when printing evaluation summary
    '''

    raw_pred, _, _ = trainer.predict(test_dataset)
    y_pred_bool = np.argmax(raw_pred, axis=1)
    if save:
        save_predictions(y_pred_bool, config)

    if print_eval:
        print("\n...............................\n")
        print("Evaluations:")
        print("\n...............................\n")
        print(classification_report(test_data["label"], y_pred_bool, digits=4))

    return y_pred_bool


def run_bert_evaluation(config):
    '''
    Evaluating the model specified in config. Either from saved pre-trained model or pre-training model from scratch
    Prints evaluations and confusion matrix
    '''
    train = load_dataframe(config)
    dev = train.sample(frac=0.2)
    test = load_dataframe(config, test=True)

    print("...........................\n")
    print("Length train: ", len(train))
    print("Length test: ", len(test))
    print("...........................\n")

    train = configure_training_set(train, config)

    train_dataset, dev_dataset, test_dataset = encode_datasets(train, dev, test, config)

    if config["trainOrEval"] == "train":
        model = fine_tune(config, train_dataset, dev_dataset)
    elif config["trainOrEval"] == "eval":
        model = load_model(config)
    else:
        raise Exception("Missing argument: trainOrEval")

    y_pred_bool = classify(config, model, test_dataset, test, print_eval=True)
    cm = confusion_matrix(test["label"], y_pred_bool)

    print("...........................\n")
    print("Confusion matrix:")
    print(cm)
    print("...........................\n")


def run_bert_cv(config, k_fold=5, summary=True):
    '''
    Running k-fold cross-validation on the training dataset
    k default to 5
    '''
    df = load_dataframe(config)

    kf = StratifiedKFold(n_splits=k_fold, random_state=seed, shuffle=True)

    f1_macros = []
    p_macros = []
    r_macros = []
    accs = []
    cms = []

    folds = 1
    for train_index, val_index in kf.split(X=df["text"], y=df["label"]):
        train = df.iloc[train_index]
        dev = train.sample(frac=0.2)
        test = df.iloc[val_index]

        train = configure_training_set(train, config)

        train_dataset, dev_dataset, test_dataset = encode_datasets(train, dev, test, config, summary=False)

        trainer = fine_tune(config, train_dataset, dev_dataset, save=False)

        y_pred_bool = classify(config, trainer, test_dataset, test, print_eval=False, save=False)

        p_macro, r_macro, f_macro, support_macro \
            = precision_recall_fscore_support(y_true=test["label"], y_pred=y_pred_bool, labels=[1, 0], average='macro')

        acc = accuracy_score(test["label"], y_pred_bool)

        f1_macros.append(f_macro)
        p_macros.append(p_macro)
        r_macros.append(r_macro)
        accs.append(acc)

        if folds == 1:
            cm = confusion_matrix(test["label"], y_pred_bool)
            cms = cm
        else:
            cm = confusion_matrix(test["label"], y_pred_bool)
            for i in range(0,len(cm)):
                for j in range(0,len(cms)):
                    cms[i][j] += cm[i][j]
        folds += 1

        if summary:
            print("Results fold", folds, ": ")
            print(classification_report(test["label"], y_pred_bool, digits=4))
            print()
            print("Confusion matrix")
            print(cm)

    print(cms)
    for i in range(len(cms)):
        for j in range(len(cms)):
            cms[i][j] = cms[i][j] / k_fold

    print("\n...............................\n")
    print("Summary of " + str(k_fold) + "-fold CV\n")

    print("Accuracy:", sum(accs) / len(accs))
    print("F1 macro:", sum(f1_macros) / len(f1_macros))
    print("Precision macro:", sum(p_macros) / len(p_macros))
    print("Recall macro:", sum(r_macros) / len(r_macros))
    print("Average confusion matrix: ")
    print(cms)
    print("\n...............................\n")

    return sum(accs) / len(accs), sum(f1_macros) / len(f1_macros), sum(p_macros) / len(p_macros), sum(r_macros) / len(r_macros), cms


def run_hyperparameter_optimization(config, search_params = {'init_lr': [5e-5, 3e-5, 2e-5], 'batch_size': [32, 16], 'num_epochs': [4, 3, 2]}):
    '''
    Hyperparameter optimization.
    Runs 5 fold cross-validation for each possible combination of hyperparameters and prints a summary of the results
    If full_optimize in config, the search corresponds to {'init_lr': [5e-5, 3e-5, 2e-5, 1e-5], 'batch_size': [32, 16, 8], 'num_epochs': [20, 15, 10, 5, 4, 3, 2]}
    If not, the search corresponds to {'init_lr': [5e-5, 3e-5, 2e-5], 'batch_size': [32, 16], 'num_epochs': [4, 3, 2]}
    '''
    max_f1 = 0
    best_f1 = []
    max_acc = 0
    best_acc = []

    if "full_optimize" in config:
        search_params = {'init_lr': [5e-5, 3e-5, 2e-5, 1e-5], 'batch_size': [32, 16, 8], 'num_epochs': [20, 15, 10, 5, 4, 3, 2]}
        if "e" in config:
            search_params['init_lr'] = [config['init_lr']]



    model = 1
    for i in range(len(search_params['init_lr'])):
        init_lr = search_params['init_lr'][i]
        for j in range(len(search_params['batch_size'])):
            batch_size = search_params['batch_size'][j]
            for k in range(len(search_params['num_epochs'])):
                num_epochs = search_params['num_epochs'][k]

                print("\n...............................\n")
                print("Model", model)
                print("Training:", "init_lr:", init_lr, 'batch_size:', batch_size, 'num_epochs:', num_epochs, "\n")

                gs_config = {'init_lr': init_lr, 'batch_size': batch_size, 'num_epochs': num_epochs, "data_dir": config["data_dir"],
                             "target_column": config['target_column'], 'text_column': config['text_column'],
                             'end_lr': config['end_lr'], "max_seq_length": config["max_seq_length"], "trainOrEval": config["trainOrEval"],
                             "translator": config["translator"], "model_name": config["model_name"], "model_folder": config["model_folder"]}

                if "train_dk" in config:
                    gs_config["train_dk"] = config["train_dk"]
                if "undersample" in config:
                    gs_config["undersample"] = config["undersample"]
                # Running 5 fold CV
                acc, f1, p, r, cm = run_bert_cv(gs_config, k_fold=5, summary=False)


                if acc > max_acc:
                    max_acc = acc
                    best_acc = [str(init_lr), str(batch_size), str(num_epochs)]

                if f1 > max_f1:
                    max_f1 = f1
                    best_f1 = [str(init_lr), str(batch_size), str(num_epochs)]
                print()
                model += 1
    print('Best model in terms of accuracy: ')
    print('Accuracy:', max_acc)
    print('init_lr:', best_acc[0], 'batch_size:', best_acc[1], 'num_epochs:', best_acc[2])
    print()
    print('Best model in terms of macro f1: ')
    print('F1 Macro:', max_f1)
    print('init_lr:', best_f1[0], 'batch_size:', best_f1[1], 'num_epochs:', best_f1[2])
    print()
