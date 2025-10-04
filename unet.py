from tensorflow.keras import layers, models, Input

def conv_block(x, filters, kernel_size=3, padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def create_unet(input_size=(128,128,1), base_filters=32):
    inputs = Input(shape=input_size)
    # Encoder
    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = conv_block(p1, base_filters*2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = conv_block(p2, base_filters*4)
    p3 = layers.MaxPooling2D((2,2))(c3)

    c4 = conv_block(p3, base_filters*8)
    p4 = layers.MaxPooling2D((2,2))(c4)

    # Bottleneck
    bn = conv_block(p4, base_filters*16)

    # Decoder
    u4 = layers.Conv2DTranspose(base_filters*8, (2,2), strides=(2,2), padding='same')(bn)
    u4 = layers.concatenate([u4, c4])
    c5 = conv_block(u4, base_filters*8)

    u3 = layers.Conv2DTranspose(base_filters*4, (2,2), strides=(2,2), padding='same')(c5)
    u3 = layers.concatenate([u3, c3])
    c6 = conv_block(u3, base_filters*4)

    u2 = layers.Conv2DTranspose(base_filters*2, (2,2), strides=(2,2), padding='same')(c6)
    u2 = layers.concatenate([u2, c2])
    c7 = conv_block(u2, base_filters*2)

    u1 = layers.Conv2DTranspose(base_filters, (2,2), strides=(2,2), padding='same')(c7)
    u1 = layers.concatenate([u1, c1])
    c8 = conv_block(u1, base_filters)

    outputs = layers.Conv2D(1, (1,1), activation='sigmoid')(c8)

    model = models.Model(inputs, outputs, name='simple_unet')
    return model
