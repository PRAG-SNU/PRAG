import os 
from tqdm.auto import tqdm
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class RAGEvaluator:
    def __init__(self, llm, retriever, checkpoint_path):
        self.llm = llm
        self.retriever = retriever
        self.criteria = [
            'Scientific Accuracy and Depth', 'Alignment with Research Objectives',
            'Source Clarity and Reliability', 'Professional Engagement', 'Information Fidelity'
        ]
        self.checkpoint_path = checkpoint_path

    def eval_answer(self, llm_question, llm_ans):
        eval_prompt =  f"""
        As a leading researcher in plant physiology with a focus on photosynthesis, you are considering using a language model to refine research ideas and guide projects. To ensure responses are of the highest quality and relevance, and to avoid score inflation, assess each response according to the following criteria:
        
        **Scoring Metrics**:
        1. **Scientific Accuracy and Depth** : Evaluate how accurately the response reflects current scientific understanding and its depth concerning photosynthesis. Score from 1 to 10, where 10 indicates perfect accuracy and depth, and 1 indicates major inaccuracies or superficial information.
        2. **Alignment with Research Objectives** : Determine how well the response aligns with specific research objectives within plant physiology and photosynthesis. Score from 1 to 10, with 10 being perfectly aligned and 1 being irrelevant.
        3. **Source Clarity and Reliability** : Assess the clarity and accuracy of the source citations used in the response, such as specific studies, data sets, or scientific theories. Rate from 1 to 10, where 10 signifies sources are cited with exceptional clarity and reliability, and 1 where citations are poor or missing.
        4. **Professional Engagement** : Evaluate the professional tone of the response and its suitability for academic discussions among experts in photosynthesis. A score of 10 suggests an expert-level dialogue, while 1 suggests a non-professional or irrelevant tone.
        5. **Information Fidelity** : Check the fidelity of the information provided by evaluating its accuracy and the absence of fabricated content not supported by source material. Score from 1 to 10, where 10 indicates high fidelity and 1 indicates significant errors or fabrications.
        
        **Evaluation Instructions**:
        1. Focus on accuracy, clarity, depth, fidelity, and professional engagement without considering response length.
        2. A score above 5 is considered good; a score of 10 indicates a flawless response.

        **Feedback Guidance**:
        After evaluating each response, provide specific suggestions for improvements based on the metrics. Focus exclusively on areas that need enhancement to help achieve higher scores in future evaluations.
        Avoid reiterating strengths or commendable aspects of the response. Concentrate on identifying and suggesting practical steps for improvement in each scoring area.
        Ensure your feedback aims to enhance the quality, clarity, and depth of future interactions with the subject to mitigate any bias and to drive meaningful advancement in the field of photosynthesis research.
            
        Question toward LLM : {llm_question}
        """
        
        eval_template = eval_prompt + """
        Retrieved Context : {context}
        LLM's Answer : {question}
        
        The format of answer will be as below: You must follow this format strictly
        ``
        Scientific Accuracy and Depth : []
        Alignment with Research Objectives : []
        Source Clarity and Reliability : []
        Professional Engagement : []
        Information Fidelity : []
        Feedback : []
        ``
        `"""

        
        QA_CHAIN_PROMPT = PromptTemplate(
            template = eval_template,
            input_variables = [
                'context',
                'question'
            ])
        
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever, chain_type_kwargs = {"prompt" : QA_CHAIN_PROMPT})
    
        response = qa_chain(llm_ans)
    
        return response['result']


    def score_parsing(self, response, criterion):
        try:
            score = int(response.split(criterion+' : [')[1][0])
        except:
            score = int(response.split(criterion+': [')[1][0])
        return score

    
    def feedback_parsing(self, response):
        try:
            feedback = response[response.index('Feedback : ')+len('Feedback : '):].strip()
        except:
            try:
                feedback = response[response.index('Feedback: ')+len('Feedback: '):].strip()
            except:
                feedback = 'none'
        return feedback


    def eval_process(self, df, iteration, check_point = False, init=False):
        df_temp = df.copy()
        col = 'base_score' if init else f'rag_score_{iteration}'
        ans_col = 'base_answer' if init else f'rag_answer_{iteration}'
      
        df_temp[col] = 'n'
        df_temp[col] = df_temp[col].astype(object)
        
        indices = list(df_temp[df_temp[col] == 'n'].index)
        
        while indices:
            for i in tqdm(indices):
                eval_result = self.eval_answer(df_temp.iloc[i]['question'], df_temp.iloc[i][ans_col])
                score_dict = {}
                try:
                    for crit in self.criteria:
                        score_dict[crit] = self.score_parsing(eval_result, crit)
                    score_dict['feedback'] = self.feedback_parsing(eval_result)
                    df_temp.at[i, col] = score_dict
                except Exception as err:
                    print(err)
                    
            indices = list(df_temp[df_temp[col] == 'n'].index)

        if check_point:
            df_temp.to_csv(self.checkpoint_path, index = False)
        else:
            pass
        
        return df_temp

    def summarize_scores(self, df, iteration, init = False):
        col = 'base_score' if init else f'rag_score_{iteration}'
        feedbacks = ''
    
        summaries = {crit: pd.Series([df[col].iloc[i][crit] for i in range(len(df))]).describe() for crit in self.criteria}
        summaries['Average Score'] = pd.Series(sum([df[col].map(lambda x: x[crit]) for crit in self.criteria])/len(self.criteria)).describe()
    
        # Filter feedback where the score is less than or equal to Q1 (which is 1 or below)
        crit_q1s = {crit : summaries[crit]['25%'] for crit in self.criteria}
        under_q1 = {crit: df[df[col].apply(lambda x: x.get(crit, 0) < crit_q1s[crit])].index.tolist() for crit in self.criteria}
        feedback_indices = set(num for sublist in under_q1.values() for num in sublist)

        print('N of Fail :',len(feedback_indices))
        for i in feedback_indices:
            feedbacks += df[col].iloc[i]['feedback']
            

        return summaries, feedbacks

    def check_iter_scores(self, df, iteration, init = False):
        col = 'base_score' if init else f'rag_score_{iteration}'

        iter_score = {crit: [df.iloc[i][col][crit] for i in range(len(df))] for crit in self.criteria}
        iter_score['Average Score'] = sum([df[col].map(lambda x: x[crit]) for crit in self.criteria])/len(self.criteria)

        print(f'Score Distribution ..................')
        print(pd.DataFrame(iter_score).describe())
        print()