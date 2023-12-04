import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from data_loader.data_loader import load_data
from preprocessing.preprocessing import prepare_dataset
from visualization.visualization import visualize_histograms
from regressors.regressors import (random_forest_regression,
                                    support_vector_machine,
                                    linear_regression,
                                    multiple_linear_regression,
                                    decision_tree_regression,
                                    polynomial_regression)


def get_data():
    dataset_name = "housing.csv"
    dataset = load_data(dataset_name)
    Xtrain, Ytrain, Xtest, Ytest, rYtrain, rYtest = prepare_dataset(dataset)
    return dataset, Xtrain, Ytrain, Xtest, Ytest, rYtrain, rYtest

def execute_algorithm(algorithm_func, Xtrain, Ytrain, Xtest, Ytest, rYtrain, rYtest):
    try:
        if algorithm_func in [random_forest_regression, support_vector_machine]:
            metrics = algorithm_func(Xtrain, rYtrain, Xtest, rYtest)
        else:
            metrics = algorithm_func(Xtrain, Ytrain, Xtest, Ytest)
        messagebox.showinfo(
            "Metrics",
            f"Explained Variance: {metrics[0]}\nR2 Score: {metrics[1]}\nMean Squared Error: {metrics[2]}\nRoot Mean Squared Error: {metrics[3]}\nMax Error: {metrics[4]}",
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def add_button(root, text, command, row, column, columnspan, pady):
    button = ttk.Button(root, text=text, command=command, style="Cyan.TButton")
    button.grid(row=row, column=column, columnspan=columnspan, pady=pady, sticky="ew")

def create_gui(root):
    root.title("Regression Algorithms")
    root.geometry("500x500")

    style = ttk.Style()
    style.configure("TButton", padding=(10, 5), font=("Helvetica", 12))
    style.configure("TLabel", font=("Helvetica", 14, "bold"))
    style.configure("Cyan.TButton", background="cyan", foreground="black")
    style.configure("Red.TButton", background="red", foreground="black")  # Add this line
    
    label = ttk.Label(root, text="Select Regression Algorithm to Execute")
    label.grid(row=0, column=0, columnspan=2, pady=(20, 10))

    dataset, Xtrain, Ytrain, Xtest, Ytest, rYtrain, rYtest = get_data()

    visualize_button = ttk.Button(root, text="Visualize Metrics", command=lambda: visualize_histograms(dataset), style="Red.TButton")
    visualize_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

    algorithms = [
        ("Linear Regression", linear_regression),
        ("Multiple Linear Regression", multiple_linear_regression),
        ("Polynomial Regression", polynomial_regression),
        ("Decision Tree Regression", decision_tree_regression),
        ("Random Forest Regression", random_forest_regression),
        ("Support Vector Machine", support_vector_machine)
    ]

    for i, (algorithm_name, algorithm_func) in enumerate(algorithms, start=2):
        add_button(root, 
                   text=algorithm_name, 
                   command=lambda func=algorithm_func: execute_algorithm(func, Xtrain, Ytrain, Xtest, Ytest, rYtrain, rYtest), 
                   row=i, 
                   column=0, 
                   columnspan=2, 
                   pady=10)

    # Centering the columns and rows
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    for i in range(1, len(algorithms) + 3):
        root.rowconfigure(i, weight=1)

def main():
    root = tk.Tk()
    create_gui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
