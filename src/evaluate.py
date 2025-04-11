# Script is still incomplete
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
import argparse
from get_data import read_params
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import uuid

def model_evaluate(config_file: str, model_path: str = '../models/trained.h5'):
    config = read_params(config_file)

    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    pred_classes = config['raw_data']['classes']

    te_set = config['model']['test_path']
    model_path = config['model']['sav_dir']

    model = load_model(model_path)

    test_gen = ImageDataGenerator(rescale = 1./255)
    test_set = test_gen.flow_from_directory(te_set,
                                                target_size = (225,225),
                                                batch_size = batch,
                                                class_mode = class_mode,
                                                shuffle=False
                                                )
    
    y_true = test_set.classes
    class_labels = list(test_set.class_indices.keys())
    print(class_labels)

    y_probs = model.predict(test_set, steps=len(test_set))
    y_pred = np.argmax(y_probs, axis=1)

    confidences = np.max(y_probs, axis=1)
    avg_confidence = float(np.mean(confidences) * 100)  

    print("Confusion Matrix")
    sns.heatmap(confusion_matrix(test_set.classes,y_pred),annot=True)
    plt.xlabel(f'Actual Values | 0:{pred_classes[0]}, 1:{pred_classes[1]}, 2:{pred_classes[2]}, 3:{pred_classes[3]}')
    plt.ylabel(f'Predicted Values | 0:{pred_classes[0]}, 1:{pred_classes[1]}, 2:{pred_classes[2]}, 3:{pred_classes[3]}')
    plt.savefig(f'reports/Confusion_Matrix_{uuid.uuid4().hex[:8]}')

    print("Classification Report")
    df =pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=pred_classes, output_dict=True)).T
    df['support']=df.support.apply(int)
    df.style.background_gradient(cmap='viridis',subset=pd.IndexSlice['0':'9','f1-score'])
    df.to_csv(f'reports/classification_report_{uuid.uuid4().hex[:8]}')
    print('Classification Report and Confusion Matrix Report are saved in reports folder of Template')

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confidence": round(avg_confidence, 2)
    }

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    passed_args=parser.parse_args()
    result  = model_evaluate(config_file=passed_args.config)
    print(result)