# autoencoder
def getModel():
    input_img = Input(shape=(48, 48, 1))
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
    #6x6x32 -- bottleneck
    x = UpSampling2D((2, 2), dim_ordering='tf')(encoded)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    x = UpSampling2D((2, 2), dim_ordering='tf')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)

    #Create model
    autoencoder = Model(input_img, decoded)
    return autoencoder

# Trains the model for 10 epochs
def trainModel():
    # Load dataset
    print("Loading dataset...")
    x_train_gray, x_train, x_test_gray, x_test = getDataset()

    # Create model description
    print("Creating model...")
    model = getModel()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

    # Train model
    print("Training model...")
    model.fit(x_train_gray, x_train, nb_epoch=10, batch_size=148, shuffle=True, validation_data=(x_test_gray, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    # Evaluate loaded model on test data
    print("Evaluating model...")
    score = model.evaluate(x_train_gray, x_train, verbose=0)
    print "%s: %.2f%%" % (model.metrics_names[1], score[1]*100)

    # Serialize model to JSON
    print("Saving model...")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    print("Saving weights...")
    model.save_weights("model.h5")
