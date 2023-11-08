<aside>
💡 Data Augmentation

</aside>

- Preparing the Data
    
    ```python
    import tensorflow.keras as keras
    import pandas as pd
    
    # Load in our data from CSV files
    train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
    valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")
    
    # Separate out our target values
    y_train = train_df['label']
    y_valid = valid_df['label']
    del train_df['label']
    del valid_df['label']
    
    # Separate our our image vectors
    x_train = train_df.values
    x_valid = valid_df.values
    
    # Turn our scalar targets into binary categories
    num_classes = 24
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    
    # Normalize our image data
    x_train = x_train / 255
    x_valid = x_valid / 255
    
    # Reshape the image data for the convolutional network
    x_train = x_train.reshape(-1,28,28,1)
    x_valid = x_valid.reshape(-1,28,28,1)
    ```
    
- Model creation
    
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense,
        Conv2D,
        MaxPool2D,
        Flatten,
        Dropout,
        BatchNormalization,
    )
    
    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                     input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_classes, activation="softmax"))
    ```
    
- Data Augmentation
    - ImageDataGenerator: image augmenatation
    
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=False, # Don't randomly flip images vertically
    )
    #vertical_flip은 안하고 horizontal_flip을 하는 이유?
    #수화는 위 아래 뒤집히면 안되지만 왼손이 쓰일 경우 좌우 반전은 있어야 하기 때문
    ```
    
- Batch Size: 각 batch에 포함되는 데이터 샘플의 수를 말함
    
    ```python
    # 효율적인 연산이 가능하고 그라디언트 계산에 유리하다.
    import matplotlib.pyplot as plt
    import numpy as np
    batch_size = 32
    img_iter = datagen.flow(x_train, y_train, batch_size)
    
    x, y = img_iter.next()
    fig, ax = plt.subplots(nrows=4, ncols=8)
    for i in range(batch_size):
    	image = x[i]
    	ax.flatten()[i].imshow(np.squeeze(image))
    plt.show()
    ```
    
- Fitting the Data to the Generator
    
    ```python
    datagen.fit(x_train)
    ```
    
- Compliling the Model
    
    ```python
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    ```
    
- Training with Augmentation
    
    ```python
    model.fit(img_iter,
    					epochs=10,
    					steps_per_epoch=len(x_train)/batch_size,
    					validation_data=(x_valid, y_valid))
    ```
    
    - steps_per_epochs: To set the number of steps equal to the non-augmented dataset size divided by the batch_size (which has a default value of 32).
    즉, 훈련 데이터를 에포크당 반복할 횟수를 나타낸다.
- Saving the Model
    
    ```python
    model.save('asl_model')
    ```
    
- Summary
    - Clear the Memory
        
        ```python
        import Ipython
        app = IPython.Application.instance()
        app.kernel.do_shutdown(True)
        ```
