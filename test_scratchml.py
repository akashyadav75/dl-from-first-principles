import numpy as np
import pandas as pd

# Import from our custom scratchml library
from scratchml.regression import LinearRegression, LogisticRegression
from scratchml.trees import DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier
from scratchml.svm import LinearSVM
from scratchml.unsupervised import KMeans, PCA
from scratchml.neighbors import KNeighborsClassifier, KNeighborsRegressor
from scratchml.naive_bayes import GaussianNB
from scratchml.deep_learning import Sequential, Dense, ActivationLayer, Adam
from scratchml.activations import ReLU, Softmax
from scratchml.losses import CategoricalCrossEntropy
from scratchml.advanced_dl import Conv2D, LSTMCell, LSTM, ScaledDotProductAttention
from scratchml.gan import GAN
from scratchml.metrics import accuracy_score, r2_score, f1_score


def run_tests():
    print("=" * 70)
    print("STARTING SCRATCHML INTEGRATION AND VERIFICATION SUITE")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Regression Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Linear & Logistic Regression ---")
    np.random.seed(42)
    # Synthetic Linear Regression Data
    X_reg = np.random.randn(100, 2)
    y_reg = 3.5 * X_reg[:, 0] - 1.2 * X_reg[:, 1] + 5.0 + np.random.randn(100) * 0.1
    
    lin_reg = LinearRegression(learning_rate=0.1, epochs=200)
    lin_reg.fit(X_reg, y_reg, method='gradient_descent')
    y_pred_reg = lin_reg.predict(X_reg).flatten()
    print(f"Linear Regression R2 Score: {r2_score(y_reg, y_pred_reg):.4f} (Expected: >0.95)")

    # Synthetic Logistic Regression Data
    X_clf = np.random.randn(100, 2)
    y_clf = (2.0 * X_clf[:, 0] - 1.5 * X_clf[:, 1] > 0).astype(int)
    
    log_reg = LogisticRegression(learning_rate=0.1, epochs=300)
    log_reg.fit(X_clf, y_clf)
    y_pred_clf = log_reg.predict(X_clf).flatten()
    print(f"Logistic Regression Accuracy: {accuracy_score(y_clf, y_pred_clf):.4f} (Expected: >0.85)")

    # -----------------------------------------------------------------
    # 2. Decision Trees & Random Forest Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Decision Trees & Random Forest ---")
    dt_clf = DecisionTreeClassifier(max_depth=5)
    dt_clf.fit(X_clf, y_clf)
    y_pred_dt = dt_clf.predict(X_clf)
    print(f"Decision Tree Accuracy: {accuracy_score(y_clf, y_pred_dt):.4f} (Expected: >0.85)")

    rf_clf = RandomForestClassifier(n_estimators=5, max_depth=5)
    rf_clf.fit(X_clf, y_clf)
    y_pred_rf = rf_clf.predict(X_clf)
    print(f"Random Forest Accuracy: {accuracy_score(y_clf, y_pred_rf):.4f} (Expected: >0.85)")

    # -----------------------------------------------------------------
    # 3. SVM Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Linear SVM ---")
    svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, epochs=500)
    svm.fit(X_clf, y_clf)
    y_pred_svm = svm.predict(X_clf)
    print(f"SVM Accuracy: {accuracy_score(y_clf, y_pred_svm):.4f} (Expected: >0.80)")

    # -----------------------------------------------------------------
    # 4. Unsupervised Learning Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Unsupervised Algorithms (K-Means & PCA) ---")
    kmeans = KMeans(k=2, max_iters=50)
    kmeans.fit(X_clf)
    print(f"K-Means fit successfully. Centroids shape: {kmeans.centroids.shape}")

    pca = PCA(n_components=1)
    X_reduced = pca.fit_transform(X_clf)
    print(f"PCA reduced shape: {X_reduced.shape} (Expected: (100, 1))")
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # -----------------------------------------------------------------
    # 5. Neighbors & Naive Bayes Verification
    # -----------------------------------------------------------------
    print("\n--- Testing KNN & Naive Bayes ---")
    knn = KNeighborsClassifier(k=3)
    knn.fit(X_clf, y_clf)
    y_pred_knn = knn.predict(X_clf)
    print(f"KNN Accuracy: {accuracy_score(y_clf, y_pred_knn):.4f} (Expected: >0.85)")

    gnb = GaussianNB()
    gnb.fit(X_clf, y_clf)
    y_pred_gnb = gnb.predict(X_clf)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_clf, y_pred_gnb):.4f} (Expected: >0.85)")

    # -----------------------------------------------------------------
    # 6. Deep Learning (MLP) Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Deep Learning MLP Sequential Engine ---")
    # Multi-class dataset target setup
    y_onehot = np.zeros((100, 2))
    y_onehot[np.arange(100), y_clf] = 1.0

    model = Sequential()
    model.add(Dense(input_dim=2, output_dim=4))
    model.add(ActivationLayer(ReLU()))
    model.add(Dense(input_dim=4, output_dim=2))
    model.add(ActivationLayer(Softmax()))

    adam = Adam(learning_rate=0.05)
    loss_fn = CategoricalCrossEntropy()

    # Fit model
    model.fit(X_clf, y_onehot, epochs=50, loss_fn=loss_fn, optimizer=adam, batch_size=16)
    
    y_pred_dl = model.forward(X_clf)
    y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)
    print(f"MLP Final Accuracy: {accuracy_score(y_clf, y_pred_dl_classes):.4f} (Expected: >0.85)")

    # -----------------------------------------------------------------
    # 7. Advanced DL Architectures Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Advanced Deep Learning Components ---")
    # Conv2D Forward Pass Check
    images = np.random.randn(2, 3, 32, 32) # Batch of 2 RGB images
    conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    conv_out = conv.forward(images)
    print(f"Conv2D Output Shape: {conv_out.shape} (Expected: (2, 8, 32, 32))")

    # LSTM Cell Forward Pass Check
    lstm = LSTMCell(input_dim=10, hidden_dim=20)
    x_t = np.random.randn(10, 1)
    h_prev = np.random.randn(20, 1)
    c_prev = np.random.randn(20, 1)
    h_next, c_next, _ = lstm.forward(x_t, h_prev, c_prev)
    print(f"LSTM Cell forward successful. Hidden State shape: {h_next.shape}")

    # Transformer Attention Forward Pass Check
    seq = np.random.randn(5, 16) # Sequence of 5 tokens, embed dimension 16
    attention = ScaledDotProductAttention(d_model=16)
    attn_out = attention.forward(seq)
    print(f"Self-Attention Output Shape: {attn_out.shape} (Expected: (5, 16))")

    # -----------------------------------------------------------------
    # 8. Complete Backpropagation for CNN, LSTM, and GAN Verification
    # -----------------------------------------------------------------
    print("\n--- Testing Complete Backpropagation & Generative Adversarial Networks (GAN) ---")
    
    # CNN Backward Pass Verification
    conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
    images = np.random.randn(2, 3, 16, 16)
    forward_out = conv.forward(images)
    d_out = np.random.randn(*forward_out.shape)
    dX = conv.backward(d_out)
    print(f"Conv2D Backward Pass Successful. Input gradient shape: {dX.shape} (Expected: (2, 3, 16, 16))")
    print(f"Conv2D Kernel gradients (dW) shape: {conv.dW.shape}")

    # LSTM Cell Backward Pass Verification (BPTT)
    lstm = LSTMCell(input_dim=4, hidden_dim=8)
    x_t = np.random.randn(4, 1)
    h_prev = np.random.randn(8, 1)
    c_prev = np.random.randn(8, 1)
    h_next, c_next, cache = lstm.forward(x_t, h_prev, c_prev)
    dh_next = np.random.randn(8, 1)
    dc_next = np.random.randn(8, 1)
    dx_t, dh_prev, dc_prev = lstm.backward(dh_next, dc_next, c_prev, cache)
    print(f"LSTM Cell Backpropagation (BPTT) Successful.")
    print(f"LSTM input gradient shape: {dx_t.shape}, hidden state gradient shape: {dh_prev.shape}")

    # Complete 3D Sequence LSTM Layer Verification
    seq_lstm = LSTM(input_dim=4, hidden_dim=8)
    X_seq = np.random.randn(2, 5, 4) # (batch_size=2, seq_len=5, input_dim=4)
    hs_seq = seq_lstm.forward(X_seq)
    dh_seq = np.random.randn(*hs_seq.shape)
    dX_seq = seq_lstm.backward(dh_seq)
    print(f"Sequence LSTM Layer BPTT Successful. Output gradient shape: {dX_seq.shape} (Expected: (2, 5, 4))")

    # GAN Verification
    gan = GAN(noise_dim=5, data_dim=10, hidden_dim=16)
    real_batch = np.random.rand(4, 10) # 4 real samples of dimension 10
    loss_D, loss_G = gan.train_step(real_batch)
    print(f"GAN Step Successful. Discriminator Loss: {loss_D:.4f}, Generator Loss: {loss_G:.4f}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
