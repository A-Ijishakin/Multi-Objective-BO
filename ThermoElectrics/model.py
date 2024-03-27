#from sklearn.gaussian_process import GaussianProcessRegressor as GPR
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.gaussian_process.kernels import ConstantKernel as CK

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

from joblib import dump,load
import numpy as np
import matplotlib.pyplot as plt

from features import get_datasets, featurize_and_targets

def hyperparam_tune(data,model):
    # Setup RandomizedSearchCV
    X_train, X_test, y_train, y_test = data[:]
    param_distributions = {
        'estimator__n_estimators': [100, 200, 300],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__max_depth': [3, 4, 5],
        'estimator__min_samples_split': [2, 4, 6],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__subsample': [0.8, 0.9, 1.0]
    }
    random_search = RandomizedSearchCV(estimator=model, 
                                    param_distributions=param_distributions, 
                                    n_iter=100, 
                                    scoring='r2', 
                                    cv=5, 
                                    verbose=2, 
                                    random_state=42, 
                                    n_jobs=-1)
    random_search.fit(X_train, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best score found: ", random_search.best_score_)
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test set R^2 score: {test_score}")

    return best_model,random_search.best_params_

def prepare_data():
    dataset = get_datasets()
    features,targets = featurize_and_targets(dataset)
    X_= features.values
    y_ = targets.values
    data = train_test_split(X_, y_, test_size=0.1, random_state=42)
    return data

def fit_gbr_model(tune=False):
    """
    Multi-output Gradient Boosting Regressor
    """
    data = prepare_data()
    training_loss = []

    if tune == True:
        gradient_boosting_regressor = GradientBoostingRegressor(random_state=42)
        multi_output_regressor = MultiOutputRegressor(gradient_boosting_regressor)
        model,_= hyperparam_tune(data, multi_output_regressor)
        dump(model,'gbr_model.joblib')
        return model
    else:
        gradient_boosting_regressor = GradientBoostingRegressor(random_state=42,
                                        learning_rate=0.1,
                                        subsample=0.8,
                                        n_estimators=500, #200,
                                        max_depth=4,
                                        min_samples_leaf=1,
                                        min_samples_split=4,
                                        )
        model = MultiOutputRegressor(gradient_boosting_regressor)
        X_train, X_test, y_train, y_test = data[:]
        model.fit(X_train,y_train)
        for y_pred_stage in model.estimators_[0].staged_predict(X_train):
            training_loss.append(mean_squared_error(y_train[:, 0], y_pred_stage))
        test_score = model.score(X_test, y_test)
        print(f"Test set R^2 score: {test_score}")
        dump(model,'gbr_model.joblib')
        return model, data, training_loss

def fit_gpr_model():
    # NOTE: Not practical given large dimension and small dataset. Will overfit!
    # Assume X_train is your input features and y_train_seebeck is the target Seebeck coefficient
    #kernel = CK(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    #gpr = GPR(kernel=kernel, n_restarts_optimizer=10)
    #multi_output_regressor = MultiOutputRegressor(gpr)
    #multi_output_regressor = gpr.fit(X_test,y_test)
    pass

def load_model(filepath):
    return load(filepath)


def generate_metric_plots(y_test, y_pred, training_loss, target_names):
    n_targets = y_test.shape[1]
    fig, axs = plt.subplots(1, n_targets + 1, figsize=(6 * (n_targets + 1), 5))

    # Generate parity plots for each target
    for i in range(n_targets):
        axs[i].scatter(y_test[:, i], y_pred[:, i], color='blue', alpha=0.5)
        axs[i].plot([y_test[:, i].min(), y_test[:, i].max()], 
                    [y_test[:, i].min(), y_test[:, i].max()], 'k--', lw=2)
        axs[i].set_xlabel('Measured')
        axs[i].set_ylabel('Predicted')
        axs[i].set_title(f'Parity Plot: {target_names[i]}')
        r2 = r2_score(y_test[:, i], y_pred[:, i])

        # Add R^2 score to the parity plots
        axs[i].text(0.05, 0.95, f'R^2: {r2:.2f}', transform=axs[i].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if training_loss:
        axs[-1].plot(training_loss, label='Training Loss', color='red')
        axs[-1].set_yscale('log')  
        axs[-1].set_xlabel('Iterations')
        axs[-1].set_ylabel('Log Mean Squared Error')
        axs[-1].set_title('Training Loss Over Iterations (Log Scale)')
        
        loss_min, loss_max = min(training_loss), max(training_loss)
        log_ticks = np.logspace(np.log10(loss_min), np.log10(loss_max), num=10)
        axs[-1].set_yticks(log_ticks)
        axs[-1].set_yticklabels(["{:.2e}".format(tick) for tick in log_ticks])
        
        axs[-1].legend()

    plt.tight_layout()
    plt.savefig('fit_metrics.png')

if __name__ == '__main__':
    model, data, loss = fit_gbr_model()
    _, X_test,_,y_test = data[:]
    y_predict = model.predict(X_test)
    names = ['Seebeck Coefficient', 'Electrical Conductivity', 'Thermal Conductivity']
    generate_metric_plots(y_test,y_predict,loss,names)
