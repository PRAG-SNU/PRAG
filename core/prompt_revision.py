from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class PromptRevisionManager:
    def __init__(self, llm):
        self.llm = llm
        self.history = []

    def revise_prompt(self, prev_prompt, prev_scores, current_scores, feedbacks):
        comparison_lines = []
        
        for criterion in prev_scores.keys():
            line = f"[{criterion}] \nPrev (mean: {prev_scores[criterion]['mean']}, std: {prev_scores[criterion]['std']}, min: {prev_scores[criterion]['min']}, q2: {prev_scores[criterion]['50%']}, max: {prev_scores[criterion]['max']}) \n-> Current (mean: {current_scores[criterion]['mean']}, std: {current_scores[criterion]['std']}, min: {current_scores[criterion]['min']}, q2: {current_scores[criterion]['50%']}, max: {current_scores[criterion]['max']})"
            comparison_lines.append(line)
        
        score_info = "Previous Guideline: " + prev_prompt + "\n" + "\n".join(comparison_lines) + "\nFeedbacks: " + feedbacks
        #print(score_info)
        
        revise_template = """
        Evaluation Results : {question}
        
        You have received feedback and performance scores from an AI research assistant's responses on a photosynthesis study. Your task is to develop guidelines that will enhance the assistant's performance for future interactions. 
        Examine areas of weakness closely and propose specific, actionable steps for overall improvement. Detailed guidance should be provided for any aspects that are particularly lacking.

        Final Format:
        'Overall Guideline:'
        """
        
        CHAIN_PROMPT = PromptTemplate(template=revise_template, input_variables=["question"])
        llm_chain = LLMChain(llm = self.llm, prompt = CHAIN_PROMPT)
        response = llm_chain(score_info)
    
        try:
            new_prompt = response['text'][len('Overall Guideline: '):].strip()
        except:
            new_prompt = response['text'][len('Overall Guideline : '):].strip()

        self.history.append(new_prompt)
        
        return new_prompt

    def get_history(self):
        return self.history