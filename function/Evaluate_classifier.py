
def train_and_evaluate_classifier(classifier, train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels):
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

    
    train_labels = train_labels.astype(int)
    val_labels = val_labels.astype(int)
    test_labels = test_labels.astype(int)
    
    # Train the classifier
    classifier.fit(train_embeddings, train_labels)
    
    # Evaluate on validation set
    val_predictions = classifier.predict(val_embeddings)
    #val_accuracy = accuracy_score(val_labels, val_predictions)
    val_f1 = f1_score(val_labels, val_predictions, average='binary') 

    
    # Evaluate on test set
    test_predictions = classifier.predict(test_embeddings)
    test_cm = confusion_matrix(test_labels, test_predictions)
    #test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions, average='binary')
    test_recall = recall_score(test_labels, test_predictions, average='binary')
    test_f1 = f1_score(test_labels, test_predictions, average='binary')
    
    # Compute AUC-ROC if the classifier has predict_proba method
    if hasattr(classifier, "predict_proba"):
        test_auc_roc = roc_auc_score(test_labels, classifier.predict_proba(test_embeddings)[:, 1])
        val_auc_score = roc_auc_score(val_labels, classifier.predict_proba(val_embeddings)[:, 1])

    else:
        test_auc_roc = None 
        val_auc_score = None 
    
    return val_auc_score, val_f1, test_cm, test_precision, test_recall, test_f1, test_auc_roc
