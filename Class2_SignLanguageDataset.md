---
<aside>
💡 Sign Language Dataset
</aside>

- Kaggle에서 dataset 이용
- Loading the Data
    - Reading in the Data
        
        ```python
        #pandas 사용
        import pandas as pd
        
        #csv format을 사용
        train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
        valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")
        ```
        
    - Exploring the Data
        
        ```python
        train_df.head()
        ```
        
        <img width="719" alt="image" src="https://github.com/Dororo99/nvidia_dli/assets/136609617/3c892af6-362a-493f-bf0c-e9e83e1cf2ef">

        
    - Extracting the labels
        
        ```python
        #x에 대한 결과값 저장하기
        y_train = train_df['label']
        y_valid = train_df['label']
        del train_df['label']
        del valid_df['label']
        ```
        
    - Extracting the Images
        
        ```python
        x_train = train_df.values
        x_valid = valid_df.values
        ```
        
- Summarizing the Training and Validation Data
    - 27,455 images with 784 pixels each for training
        
        ```python
        x_train.shape
        y_train.shape
        x_valid.shape
        y_valid.shape
        ```
        
- Visualizing the Data
    
    ```python
    import matplotlib.pyplot as plt
    plt.figure(figsize=(40,40))
    
    num_images = 20
    for i in range(num_images):
        row = x_train[i]
        label = y_train[i]
        
        image = row.reshape(28,28)
        plt.subplot(1, num_images, i+1)
        plt.title(label, fontdict={'fontsize': 30})
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    ```
