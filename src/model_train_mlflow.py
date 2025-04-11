import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from glob import glob
import os
import argparse
from get_data import read_params
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
import tensorflow as tf
import mlflow
from urllib.parse import urlparse
import mlflow.keras
from model_train import add_uuid_to_filename, add_uuid_and_suffix_to_filename

def create_data_generators(config):
    img_size = tuple(config['model']['image_size'])
    rescale = config['img_augment']['rescale']
    shear_range = config['img_augment']['shear_range']
    zoom_range = config['img_augment']['zoom_range']
    horizontal_flip = config['img_augment']['horizontal_flip']
    vertical_flip = config['img_augment']['vertical_flip']
    brightness_range = config['img_augment']['brightness_range']
    class_mode = config['img_augment']['class_mode']
    batch = config['img_augment']['batch_size']
    train_path = config['model']['train_path']
    test_path = config['model']['test_path']

    print(type(batch))

    train_gen = ImageDataGenerator(rescale = rescale, 
                                       shear_range = shear_range, 
                                       zoom_range = zoom_range, 
                                       horizontal_flip = horizontal_flip, 
                                       vertical_flip = vertical_flip,
                                       rotation_range = 90,
                                       brightness_range = brightness_range)
    test_gen = ImageDataGenerator(rescale = rescale)

    train_set = train_gen.flow_from_directory(train_path,
                                                  target_size = img_size,
                                                  batch_size = batch,
                                                  class_mode = class_mode)
    test_set = test_gen.flow_from_directory(test_path, 
                                                target_size=img_size,
                                                batch_size = batch,
                                                class_mode = class_mode)
    
    return train_set, test_set

def train_model_mlflow(config_file):
    config = read_params(config_file)
    train = config['model']['trainable']

    if train == False:
        print("Model is not trainable")
        return
    
    num_cls = config['load_data']['num_classes']
    img_size = config['model']['image_size']
    loss = config['model']['loss']
    optimizer = config['model']['optimizer']
    metrics = config['model']['metrics']
    epochs = config['model']['epochs']
    model_path = config['model']['sav_dir']

    print("[INFO] Starting initial training...")

    base_model = VGG19(input_shape=img_size + [3], weights = 'imagenet', include_top = False)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    output = Dense(num_cls, activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs = output)

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)   
    print(model.summary())

    train_set, test_set = create_data_generators(config)


    ################# START OF MLFLOW #################

    mlflow_config = config['mlflow_config']
    remote_server_uri = mlflow_config['remote_server_uri']
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config['experiment_name'])
    with mlflow.start_run():
        history = model.fit(train_set,
                          epochs = epochs,
                          validation_data = test_set,
                          steps_per_epoch = len(train_set),
                          validation_steps = len(test_set)) 
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("loss", loss)
        mlflow.log_param("val_loss", val_loss)
        mlflow.log_param("val_accuracy", val_acc)
        mlflow.log_param("metrics", acc)

        tracking_url_type_Store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_Store != "file":
            mlflow.keras.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.keras.log_model(model, "model")

    model.save(add_uuid_to_filename(model_path))
    print("[INFO] Model Saved Successfully....!")


def fine_tune_model_mlflow(config_file):
    config = read_params(config_file)

    model_path = config['model']['sav_dir']

    # Load saved model
    model = load_model(model_path)
    print("[INFO] Loaded model for fine-tuning from:", model_path)

    for layer in model.layers[-4:]:
        layer.trainable = True
    
    image_size = tuple(config['model']['image_size'])
    loss = config['model']['loss']
    metrics = config['model']['metrics']
    # optimizer = config['model']['optimizer']
    fine_tune_epochs = config['model']['fine_tune_epochs']

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=metrics
        # optimizer=optimizer
    )
    print(model.summary())

    train_set, test_set = create_data_generators(config)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1)
    ]

    ################# START OF MLFLOW #################

    mlflow_config = config['mlflow_config']
    remote_server_uri = mlflow_config['remote_server_uri']
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config['experiment_name'])
    with mlflow.start_run():
        history = model.fit(
                            train_set,
                            epochs=fine_tune_epochs,
                            validation_data=test_set,
                            steps_per_epoch=len(train_set),
                            validation_steps=len(test_set),
                            callbacks=callbacks
                            )
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        mlflow.log_param("model_name", config["model"]["name"])
        mlflow.log_param("epochs", config["model"]["epochs"])
        mlflow.log_param("loss", config["model"]["loss"])
        mlflow.log_param("optimizer", config["model"]["optimizer"])

        mlflow.log_metric("accuracy", acc)         
        mlflow.log_metric("val_accuracy", val_acc)     
        mlflow.log_metric("val_loss", val_loss)         
        mlflow.log_metric("loss", train_loss)        

        tracking_url_type_Store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_Store != "file":
            mlflow.keras.log_model(model, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.keras.log_model(model, "model")

    fine_tuned_model_path = add_uuid_and_suffix_to_filename(model_path, suffix="finetuned")
    model.save(fine_tuned_model_path)
    print(f"[INFO] Fine-tuned model saved successfully at: {fine_tuned_model_path}")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    passed_args=parser.parse_args()

    config = read_params(passed_args.config)
    mode = config.get("run_config", {}).get("mode", "train")
    
    if mode == "train":
        train_model_mlflow(config_file=passed_args.config)
    elif mode == "fine_tune":
        fine_tune_model_mlflow(config_file=passed_args.config)
    else:
        print("[ERROR] Invalid mode in config. Use 'train' or 'fine_tune'.")
