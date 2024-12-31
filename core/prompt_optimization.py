import pandas as pd
from tqdm.auto import tqdm
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pytz import timezone

from utils.config import DATA_PATH, DB_DIRECTORY, MODEL_NAME
from utils.data_repository import DataRepository
from core.rag_assistant import RAGAssistant
from core.rag_evaluator import RAGEvaluator
from core.prompt_revision import PromptRevisionManager

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class PromptOptimization:
    def __init__(self, check_point = False):
        self.data_repo = DataRepository(DATA_PATH)
        
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
        retriever = vectordb.as_retriever()

        self.Assistant = RAGAssistant(llm, retriever, checkpoint_path=f"{DATA_PATH}/temp_checkpoint.csv")
        self.Evaluator = RAGEvaluator(llm, retriever, checkpoint_path=f"{DATA_PATH}/temp_checkpoint.csv")
        self.PromptRevisor = PromptRevisionManager(llm)
        
        self.current_prompt = ''
        self.current_scores = {}
        self.prompt_history = [self.current_prompt, ]  # Initialize the history with the initial prompt

        self.check_point = check_point


    def Initialize_BaseQA(self, df):
        df_temp = df.copy()
        
        print(f"########################\n  Initialize : Base QA\n########################")

        #Evaluation
        print('Evaluation ..................')
        df_temp = self.Evaluator.eval_process(df_temp, '', self.check_point, True)
        self.current_scores, _ = self.Evaluator.summarize_scores(df_temp, '', True)
        self.Evaluator.check_iter_scores(df_temp, '', True)

        df_train = df_temp[df_temp['train_test'] == 'train'].reset_index(drop = True)
        df_test = df_temp[df_temp['train_test'] == 'test'].reset_index(drop = True)

        return df_train, df_test


    def run_iterations(self, df, num_iterations):
        df_temp = df.copy()
        
        for iteration in range(1, num_iterations + 1):
            print(f"########################\n       Iteration {iteration}\n########################")
            print(f"Prompt : \n{self.current_prompt}\n")
            
            prev_scores = self.current_scores

            # Inference
            print('Inference ..................')
            df_temp = self.Assistant.inference(df_temp, self.current_prompt, iteration, self.check_point)

            # Evaluation
            print('Evaluation ..................')
            df_temp = self.Evaluator.eval_process(df_temp, iteration, self.check_point)
            self.current_scores, feedbacks = self.Evaluator.summarize_scores(df_temp, iteration)
            self.Evaluator.check_iter_scores(df_temp, iteration)

            # Prompt Revision and History Update
            if iteration < num_iterations:
                new_prompt = self.PromptRevisor.revise_prompt(self.current_prompt, prev_scores, self.current_scores, feedbacks)
                self.current_prompt = new_prompt
                self.prompt_history.append(new_prompt)  # Save each new prompt to history
                
            # Update previous scores for the next iteration
            prev_scores = self.current_scores

        return df_temp


    def final_evaluation(self, df_test):
        df_temp = df_test.copy()
        print(f"########################\n    Final Evaluation\n########################")
        print(f"Prompt : \n{self.current_prompt}\n")

        # Inference
        print('Inference ..................')
        df_temp = self.Assistant.inference(df_temp, self.current_prompt, 'eval', self.check_point)

        # Evaluation
        print('Evaluation ..................')
        df_temp = self.Evaluator.eval_process(df_temp, 'eval')
        current_scores, feedbacks = self.Evaluator.summarize_scores(df_temp, 'eval')
        self.Evaluator.check_iter_scores(df_temp, 'eval')

        return df_temp

    def get_current_prompt(self):
        return self.current_prompt

    def get_prompt_history(self):
        return self.prompt_history
        
    def run(self, df, num_iterations, save_data = False):
        df_train, df_test = self.Initialize_BaseQA(df)
        final_train_df = self.run_iterations(df_train, num_iterations)
        final_test_df = self.final_evaluation(df_test)
        prompt_history = self.get_prompt_history()

        runtime = datetime.now(timezone('Asia/Seoul')).strftime('%m%d_%H%M')
        #print(runtime)
        if save_data:
            self.data_repo.save_data(final_train_df, filename=f'photosyn_qa_dataset_train_final_{runtime}.pickle')
            self.data_repo.save_data(final_test_df, filename=f'photosyn_qa_dataset_test_final_{runtime}.pickle')
            self.data_repo.save_data(prompt_history, filename=f'photosyn_qa_prompt_history.pickle')

        return final_train_df, final_test_df, prompt_history