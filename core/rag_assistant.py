import pandas as pd
from tqdm.auto import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class RAGAssistant:
    def __init__(self, llm, retriever, checkpoint_path):
        self.llm = llm
        self.retriever = retriever
        self.checkpoint_path = checkpoint_path

    def inference(self, df, prompt, iteration, checkpoint=False, init=False):
        df_temp = df.copy()
        result_col = 'base_answer' if init else f'rag_answer_{iteration}'
        source_col = f'rag_source_{iteration}'
        
        rag_template = prompt + """    
        Retrieved Context: {context}
        Question: {question}
        Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate(template=rag_template, input_variables=['context', 'question'])
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            retriever=self.retriever, 
            return_source_documents = True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        for question in tqdm(df_temp['question']):
            response = qa_chain(question)
            df_temp.loc[df_temp['question'] == question, result_col] = response['result']
            df_temp.loc[df_temp['question'] == question, source_col] = response['source_documents'][0].metadata['source']
            
            if checkpoint:
                df_temp.to_csv(self.checkpoint_path, index=False)
                
        return df_temp