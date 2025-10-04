import tensorflow as tf
from tensorflow.keras import layers, models

def build_efficientnet_classification(input_shape=(224,224,3), num_classes=2, trainable_backbone=False):
    base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = trainable_backbone
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out, name='efficientnet_classifier')
    return model
