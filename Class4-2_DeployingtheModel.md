<aside>
💡 Deploying your Model

</aside>

1. Load an already-trained model from disk
2. Reformat images for a model trained on images of a different format
3. Perform inference with new images, never seen by the trained model and evaluate its performance

- Loading the Model
    
    ```python
    from tensorflow import keras
    model = keras.models.load_model('asl_model')
    #load_model을 하게되면 cpu가 아닌 gpu로 바로 load하게 된다
    
    #불러와졌는지 확인하려면
    model.summary()
    ```
    
    - cpu_memory ↔ gpu_memory를 줄여야 더 빠르게 동작하도록 만들 수 있다.
- Preparing an Image for the Model
    - Showing the Images
        
        ```python
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        def show_image(image_path):
        	image = mpimg.imread(image_path)
        	plt.imshow(image, cmap='gray')
        
        #이미지를 불러오자
        show_image('data/asl_images/b.png')
        ```
        
        <img width="465" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/bbc3a948-9ca4-43e3-bdf2-e5a11d76d58c">

        
    - Scaling the Images
    이미 학습된 모델에 맞춰서 scaling을 해줘야 함!
    → grayscale & 28x28
        
        ```python
        from tensorflow.keras.preprocessing import image as image_utilss
        
        def load_and_scale_image(imgae_path):
        	image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28,28))
        	return image
        
        # 결과물을 다시 띄워보자
        image = load_and_scale_image('data/asl_images/b.png')
        plt.imshow(image, cmp='gray')
        ```
        
        <img width="515" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/05551edd-c595-49b5-8ddc-2050613216ef">

        
    - Preparing the Image for Prediction
        1. Convert our image into a more rudimentary format → Use keras utility called “image_to_array”
            
            ```python
            image = image_utils.img_to_array(image)
            ```
            
        2. Reshape our image to match the shape of the dataset the model was trained on
            
            ```python
            # 한 개의 컬러 채널과 28x28의 픽셀
            image = image.reshape(1,28,28,1)
            ```
            
        3. Normalize our data as we did with our training dataset
            
            ```python
            image = image / 255
            ```
            
- Making Predictions
    
    ```python
    prediction = model.predict(image)
    print(prediction)
    #아래 값을 보면 prediction[1]이 1인것을 볼 수 있다.
    ```
    
    <img width="602" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/a54073af-1f0d-4173-9045-901a30f6d735">

    
    - Understanding the Prediction
        
        24 length array ⇒ binarized categorical arrays from y_train and y_test와 같은 형식이다.
        → 각 배열의 원소의 확률은 [0,1]
        
        ```python
        import numpy as np
        np.argmax(prediction)
        
        #일단 j와 z를 제외한 나머지 알파벳을 순서대로 입력시켜주자
        alphabet = "abcdefghiklmnopqrstuvwxy"
        
        #prediction에 맞는 알파벳을 출력하자
        alphabet[np.argmax(prediction)]
        ```

   
   <aside>
💡 실습

</aside>

<img width="652" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/f0d6aa08-a5ea-4376-8589-7c9af157ffd6">
