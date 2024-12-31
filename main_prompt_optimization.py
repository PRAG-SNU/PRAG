from utils.config import DATA_PATH
from utils.data_repository import DataRepository
from modules.prompt_optimization import PromptOptimization

if __name__ == "__main__":
    # Initialize data repository and load initial dataset
    data_repo = DataRepository(DATA_PATH)
    df = data_repo.load_data('photosyn_qa_dataset.pickle')

    # Initialize prompt optimization and run the pipeline
    prompt_optimizer = PromptOptimization()
    final_train_df, final_test_df, prompt_history = prompt_optimizer.run(df, num_iterations=10, save_data = True)