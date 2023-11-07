---

<aside>
💡 Convolutional Neural Networks

</aside>

- Loading and Preaparing  the Data
    
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
    
    # Separate out our image vectors
    x_train = train_df.values
    x_valid = valid_df.values
    
    # Turn our scalar targets into binary categories
    num_classes = 24
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)
    
    # Normalize our image data
    x_train = x_train / 255
    x_valid = x_valid / 255
    ```
    
- Reshaping Images for a CNN
    
    ```python
    x_train.shpe, x_valid.shape
    ```
    
    - 어떤 color channels를 사용하느냐에 따라 달라지는데 현재 gray_scales를 사용중이므로 color channels의 값은 1이다.
    - 또한, 784가 아닌 28 x 28로 사용하기 위해 입력해준다
    
    ```python
    x_train = x_train.reshape(-1,28,28,1)
    x_valid = x_valid.reshape(-1,28,28,1)
    # -1을 사용하면 이전에 사용하던 값을 그대로 사용 가능하다
    # 즉, 27455 그대로 사용하려면 -1
    x_train.shape
    x_valid.shape
    x_train.shape, x_valid.shape
    ```
    
- Creating a Convolutional Model

    ![image](https://github.com/Dororo99/nvidia_dli/assets/136609617/cf7abc7d-ccba-44c0-b6c4-46dad1bd6439)

        
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
    model.add(MaxPool2D((2,2), strides=2, padding="same"))
    model.add(Conv2D(50, (3,3), strides=1, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2), strides=2, padding="same"))
    model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_classes, activation="softmax"))
    ```
    
    - Conv2D
        
        <img width="565" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/7accf9ad-209c-4162-a3fe-e53972668b34">

        
        ```python
        model.add(Conv2D(75, (3,3), strides=1, padding="same", activation="relu", input_shape=(28,28,1)))
        #75: number of filters that will be learned -> 각 필터는 이미지에서 특정 패턴 또는 특징 감지에 사용됨
        #(3,3): size of those filters
        #strides: the step size that the filter will take as it passes over the img
        #padding: same or valid
        	#same: 컨볼루션 레이어의 출력 크기를 입력 크기와 동일하게 유지하려는 옵션 -> 정보 손실 최소화 및 출력 크기 일정하게 유지
        	#valid: 컨볼루션 레이어의 출력 크기를 입력 크기보다 작게 만드는 옵션
        ```
        
    - BatchNormalization
        
        scales the values in the hidden layers to improve training
        
    - MaxPool2D
        
        <img width="562" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/e023799f-ff43-46a1-8c78-952ca02abe31">

        
        essentially shrinks it to a lower resolution
        → help the model be robust to translation and also makes our model faster
        
    - Dropout
        
        <img width="568" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/cd164dbf-2b27-4a52-aa78-11587d5f9ba4">

        
        Is to prevent overfitting
        
        ⇒ Dropout randomly selects a subset of neurons and turns them off (랜덤하게 선택하는게 포인트!) so that they dont participate in forward or backward propagation in that particular pass
        
    - Flatten
        
        Make the ouput of one layer which is multi-dimensional into a one-dimensional array(말 그대로 하나의 고차원 레이어를 일차원 레이어로 만드는것)
        
    - Dense
        
        
- Summary
    
    <img width="553" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/532db521-c33d-4bb4-b19c-e662596bf201">

    
    ```python
    model.summary()
    ```
    
- Compiling the Model
    
    ```python
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    ```
    
- Training the Model
    
    <img width="616" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/24e46e97-2269-4d65-a8ea-d6228eea9171">
