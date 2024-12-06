class Grid:
    def grid_search(classifier, param_grid, X_train, y_train, X_test, y_test, is_multi_class:bool = False, cv:int = 5):
        scoring = metric_general = [
            'precision',
            'recall',
            'f1',
            'accuracy'
        ]
        if is_multi_class:
            scoring = metric_general = {
                'precision': make_scorer(precision_score, average='micro'),
                'recall': make_scorer(recall_score, average='micro'),
                'f1': make_scorer(f1_score, average='micro'),
            }

        pipe = Pipeline([
            ('scaler', None),
            ('quantile_transformer', None),
            ('dim_reduction', None),
            ('feature_selection', None),
        ])

        pipe.steps.append(('classifier', classifier))

        param_grid.update({
            'scaler': [None, RobustScaler()],
            'quantile_transformer': [None, QuantileTransformer()],
            'dim_reduction': [None, PCA(n_components=5)],
            'feature_selection': [None, VarianceThreshold(threshold=0.1)],
        })

        grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scoring, refit = 'f1', verbose = 10, n_jobs=-1, cv = cv)
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
        print('Best Parameters:', grid.best_params_)
        print('Best Score:', grid.best_score_)
        print(classification_report(y_test, grid_predictions))
        return pd.DataFrame(grid.cv_results_)