import os

def create_project_structure():
    root_dir = 'Basic-Neural-Networks'
    sub_dirs = [
        'src/perceptron/code',
        'src/perceptron/data',
        'src/multilayer_perceptron/code',
        'src/multilayer_perceptron/data',
        'tests/perceptron_tests',
        'tests/multilayer_perceptron_tests',
        'documentation',
        'visualizations/perceptron_visualizations',
        'visualizations/multilayer_perceptron_visualizations'
    ]

    # Create directories
    os.makedirs(root_dir, exist_ok=True)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(root_dir, sub_dir), exist_ok=True)

    # Create files
    open(os.path.join(root_dir, 'LICENSE'), 'a').close()
    with open(os.path.join(root_dir, 'README.md'), 'a') as readme:
        readme.write("# Basic Neural Networks\n\n")
        readme.write("This repository contains basic neural network projects including Perceptrons and Multi-layer Perceptrons.")
    open(os.path.join(root_dir, '.gitignore'), 'a').close()
    open(os.path.join(root_dir, 'documentation/perceptron.md'), 'a').close()
    open(os.path.join(root_dir, 'documentation/multilayer_perceptron.md'), 'a').close()

if __name__ == "__main__":
    create_project_structure()