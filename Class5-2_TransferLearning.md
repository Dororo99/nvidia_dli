<aside>
💡 Transfer Learning

</aside>

1. Prepare a pretrained model for transfer learning
2. Perform transfer learning with your own small dataset on a pretrained model
⇒ 현재 상황은 Bo라는 강아지의 사진이 몇장 없음…
3. Further fine tune the model for even better performance

- A Personalized Doggy Door
    
    ⇒ Detect many different types of features that could be valuable in image recognition.
    
    사진이 “Bo”라는 강아지인지 아닌지의 여부를 판단하는것을 주된 목표로 함
    
    - include_top=False: 가장 마지막 레이어를 제거
        
        ```python
        from tensorflow import keras
        
        base_model = keras.applications.VGG16(
        	weights='imagenet',
        	input_shape=(224, 224, 3), # 여기서 3이면 rgb를 의미함
        	include_top=False) # 마지막 레이어 다운로드 안함
        
        # 결과물 출력
        model.summary()
        ```
        
        <img width="556" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/747f5536-1977-4841-b8d1-8b1d5197ceff">

        
- Freezing the Base Model
    - 기존에 있던 레이어를 업데이트 시키는 것이 아니라 새로운 레이어를 추가하는 것이기 때문에 기존의 레이어를 freeze 시켜야 한다.
        
        ```python
        base_model.trainable = False
        ```
        
- Adding New Layers
    1. Pooling layer like in convolutional neural network
    2. Add final layer (which calssify Bo or not Bo)
        
        ```python
        inputs = keras.Input(shape(224, 224, 3))
        # 기존의 모델의 가중치를 고정하고 새로운 모델의 훈련 중에 업데이트 되지 않도록 하자
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        # A Dense classifier with a single unit
        outputs = keras.layers.Dense(1)(x) # 여기서 1개의 노드면 충분한 이유?
        # Bo인지 아닌지를 구분만 하면 되기 때문에 node는 1개면 충분하다
        model = keras.Model(inputs, outputs)
        
        #결과물 보자
        model.summary()
        ```
        
        <img width="570" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/4513bfcc-c50a-4227-a63c-8d97b33aad1d">

        
- Compliling the Model
    - from_logits=True: loss function한테 아직 normalized 되지 않은 값임을 알려준다
        
        ```python
        model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])
        ```
        
- Augmenting the Data
    
    ```python
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Create a data generator
    datagen_train = ImageDataGenerator(
    	samplewise_center=True, # 각 샘플의 평균을 0으로 만들자
    	rotation_range=10,
    	zoom_range=0.1,
    	width_shift_range=0.1,
    	height_shift_range=0.1,
    	horizontal_flip=True,
    	vertical_flip=False,
    )
    
    # validation data는 augment 할 필요가 없음
    datagen_valid = ImageDataGenerator(samplewise_center=True)
    ```
    
- Loading the Data
    - flow_from_directory: Load images directly from folders
        
        ```python
        # load and iterate training dataset
        train_it = datagen_train.flow_from_directory(
        	"data/presidential_doggy_door/train/",
        	target_size=(224,224),
        	color_mode="rgb",
        	class_mode="binary",
        	batch_size=8,
        )
        
        # load and iterate validation dataset
        valid_it = datagen_valid.flow_from_directory(
        	"data/presidential_doggy_door/valid/"
        	target_size=(224,224),
        	color_mode="rgb",
        	class_mode="binary",
        	batch_size=8,
        )
        ```
        
- Training the Model
    - steps_per_epoch
        
        ```python
        model.fit(train_it, steps_per_epoch=12, validation_data=valid_it, validation_steps=4, epochs=20)
        ```
        
- Discussion of Results
    - accuracy가 1이 나왔음!!!
    이게 가능한 수치인가?!?!
- Fine-Tuning the Model
    - 주로 하는 실수?
        - small-step size를 사용해야 하는데 그렇지 않아 문제가 발생하는 경우가 많다.
    - backpropagation 사용하자
        
        ```python
        # Unfreeze the base model
        base_model.trainable=True
        
        # 업데이트 후에 꼭 recompile!!!
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
        							loss=keras.losses.BinaryCrossentropy(from_logits=True),
        							metrics=[keras.metrics.BinaryAccuracy()])
        # model fit
        model.fit(train_it, steps_per_epoch=12, validation_data=valid_it, validation_stpes=4, epochs=10)
        # validation steps: 검증 데이터를 몇 번 반복할 것인지?
        ```
        
- Examining the Predictions
    
    ```python
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from tensorflow.keras.preprocessing import image as image_utils
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    
    def show_image(image_path):
    	image = mpimg.imread(image_path)
    	plt.imshow(image)
    
    def make_predictions(image_path):
    	show_image(image_path)
    	image = image_utils.load_img(image_path, target_size=(224,224))
    	image = image_utils.img_to_array(image)
    	image = image.reshape(1,224,224,3)
    	image = prprocess_input(image)
    	preds = model.predict(image)
    	return preds
    ```
    
    <img width="651" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/4a04eec8-caa0-4ca2-81a0-d15d068c3425">

    
    <img width="686" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/33e670d6-dd94-4dd0-bb18-1af33559b587">

    
    아주 올바르게 분류된 결과를 볼 수 있다
    

<aside>
💡 실습

</aside>

```python
def presidential_doggy_door(image_path):
	preds = make_predictionos(image_path)
	if preds[0] < 0:
		print("BO")
	else: 
		print("Not Bo")
```

<img width="659" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/213f4f71-392c-4ee8-b6c3-65415ed4325e">


<img width="686" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/01dfd07c-573a-4896-88f1-028b4a9d51bb">
