<aside>
💡 Pre-Trained Models

</aside>

1. Use keras to load a very well-trined pretrained model
2. Preprocess your own images to work with the pretrained model
3. Use the pretrained model to perform accurate inference on your own images

<aside>
💡 DeepLearning

</aside>

- 2006-Sicence지
    - Ruslan, Hinton
- 2012-NeurIPS
    - NeurIPS

- An Automated Doggy Door
조건
    1. 강아지만 나갔다 들어올 수 있음
    2. 고양이는 계속 내부에만 있어야 함
    3. 나머지 동물들은 전부 밖에만 있어야 함
- Loading the Model
    - ImageNet Challenge로 부터 이미 학습된 imageNet model을 다운로드
    
    ```python
    from tensorflow.keras.applications import VGG16
    
    model = VGG16(weights="imagenet")
    ```
    
    - 결과물을 보자
    
    ```python
    model.summary()
    ```
    
    <img width="509" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/e512c6f2-8352-476a-8c0f-71200b6f47a8">

    
    - Input dimensions
        
        ```python
        # 224 pixels high, 224 pixels wide, and 3 color channels
        (224, 224, 3)
        
        # If we pass in just one image
        (1, 224, 224, 3)
        ```
        
    - Output dimensions
        
        ```python
        # predictions (Dense)에 대한 값이 (None, 1000)
        # 1000개의 카테고리로 output을 분류한것
        ```
        
- Loading an Image
    
    ```python
    import matplotlib.pyplot as plt
    import matplotlib.images as mpimg
    
    def show_image(image_path):
    	image = mpimg.imread(image_path)
    	print(image.shape)
    	plt.imshow(image)
    
    #show img
    show_image("data/doggy_door_images/happy_dog.jpg")
    ```
    
    <img width="562" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/cf17b35d-00a8-431a-af72-3579ab955759">

    
- Preprocessing the Image
    
    최종 이미지의 shape은 (1, 224, 224, 3)이 되어야 함
    
    - preprocess_imput method: 
    Allow users to preprocess their own images to match the qualities of the images that the model was originally trained on.
        
        ```python
        from tensorflow.keras.preprocessing import image as image_utils
        from tensorflow.keras.applications.vgg16 import process_input
        
        def load_and_process_image(image_path):
        	# Original shape
        	print('Original: ', mpimg.imread(image_path).shape)
        	
        	# Load in the image with a target size of 224, 224
        	image = image.utils.load_img(image_path, target_size=(224, 224))
        	# Convert the image from a PIL format to a numpy array
        	image = image_utils.img_to_array(image)
        	# Add a dimension for number of images, in our case 1
        	image = image.reshape(1, 224, 224, 3)
        	# Preprocess image to align with original ImageNet dataset
        	image = preprocess_input(image)
        	# Print image's shape after processing
        	print('Processed image shape: ', image.shape)
        	return image
        
        # 결과물 한번 보자
        processed_image = load_and_process_image("data/doggy_door_images/brown_bear.jpg")
        
        ```
        
- Make a Prediction
    - 1000개의 원
        
        ```python
        from tensorflow.keras.applications.vgg16 import decode_predictions
        
        def readable_prediction(image_path):
            # Show image
            show_image(image_path)
            # Load and pre-process image
            image = load_and_process_image(image_path)
            # Make predictions
            predictions = model.predict(image)
            # Print predictions in readable form
            print('Predicted:', decode_predictions(predictions, top=3))
        ```
        
        ```python
        readable_prediction("data/doggy_door_images/happy_dog.jpg")
        ```
        
        <img width="590" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/de07fdb0-aeec-41da-8e32-7532b3d0cbbf">

        
        ```python
        readable_prediction("data/doggy_door_images/brown_bear.jpg")
        ```
        
        <img width="599" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/b3d127c8-b968-47eb-beba-e335c1182a60">

        
        ```python
        readable_prediction("data/doggy_door_images/sleepy_cat.jpg")
        ```
        
        <img width="578" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/b4f69129-c684-4f3d-9acf-9128c1d29a25">

        
- Only Dogs
    - argmax의 값은 dogs: 151 ~ 268 cats: 281 ~ 285
        
        ```python
        import numpy as np
        
        # np.argmax: axis 중 최댓값을 리턴해줌
        
        def doggy_door(image_path):
            show_image(image_path)
            image = load_and_process_image(image_path)
            preds = model.predict(image)
            if 151<= np.argmax(preds) <= 168:
                print("Doggy come on in!")
            elif 281<= np.argmax(preds) <= 285:
                print("Kitty stay inside!")
            else:
                print("You're not a dog! Stay outside!")
        ```
        
        <img width="604" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/116e5110-c669-49f5-81e7-feb91eae3705">

        
        ⇒ 아직 정확도가 낮은것으로 보인다…
        조금 더 높일 필요성이 있어보임
