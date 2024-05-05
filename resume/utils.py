def extract_skills_from_pdf(pdf_path : str) -> str: 
    import spacy 
    import PyPDF2
    with open(pdf_path , 'rb') as pdf : 
        reader = PyPDF2.PdfReader(pdf, strict=False)
        pdf_text = []
        
        for page in reader.pages:
            content=page.extract_text()
            pdf_text.append(content)
            
        pdf_text = "\n".join(pdf_text)
    
    
    ''' YOUSEF's OLD MODEL'''
    # nlp = spacy.load("en_core_web_lg")
    # skills = r"E:\vs code\Graduation Project\Deployment\Final\jz_skill_patterns.jsonl"
    # ruler = nlp.add_pipe("entity_ruler",before='ner')
    # ruler.from_disk(skills)

    # doc = nlp(pdf_text)
    # skills_ents =[ent.text for ent in doc.ents if ent.label_=="SKILL"]
    # return skills_ents 


    ''' YOUSEF's NEW MODEL'''
    from spacy.tokens import DocBin
    from tqdm import tqdm
    import json
    # import spacy
    # from google.colab import drive
    # drive.mount('/content/drive')

    nlp = spacy.load(r"Models\spacy_cv\model-last")
    doc = nlp(pdf_text)

    s = doc.ents
    return s


# def sentence_embeddign4_similarity_score(skills):
#   import pandas as pd
#   import numpy as np 
#   from sklearn.metrics.pairwise import cosine_similarity
#   from sentence_transformers import SentenceTransformer

#   df = pd.read_csv("new_embedded_data.csv")
#   model4 = SentenceTransformer('thenlper/gte-base')


#   '**input skills should be a list**'
#   skills = [str(item) for item in skills]
#   new_skills = ', '.join(skills)

#   # Embedding the new skills
#   skills_embeddings = model4.encode(new_skills).reshape(1, -1)

#   # Calculate cosine similarity
#   df['similarity_score'] = df['skills_embeddings'].apply(lambda x: cosine_similarity([x], skills_embeddings)[0][0])

#   # Sort the DataFrame based on similarity score
#   df_sorted = df.sort_values(by='similarity_score', ascending=False)

#   # Print the top 5 most similar job titles and their similarity scores
#   print("Top 5 most similar job titles:")
#   for index, row in df_sorted.head(5).iterrows():
#       print(f"Job Title: {row['Job Title']}, Similarity Score: {row['similarity_score']}")




def sentence_embedding4_similarity_score_test(skills):
    import numpy as np
    import pandas as pd
    from sentence_transformers.util import cos_sim
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer, util
    
    l=[]
    for x in skills:
        l.append(str(x))

    new_skills = ', '.join(l)

    df = pd.read_csv('cvs/new_embedded_data.csv')

    model4 = SentenceTransformer('thenlper/gte-base')

    # Embedding the new skills
    skills_embeddings = model4.encode(new_skills).reshape(1, -1)

    # Convert string representations to NumPy arrays
    df['skills_embeddings'] = df['skills_embeddings'].apply(lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=' '))

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(np.vstack(df['skills_embeddings']), skills_embeddings)    
    

    # # Get indices of top 5 most similar job titles
    # top_indices = np.argsort(similarity_scores, axis=0)[-5:][::-1].flatten()

    # # Print top 5 most similar job titles and their similarity scores
    # print("Top 5 most similar job titles:")
    # for index in top_indices:
    #     job_title = df['Job Title'][index]
    #     similarity_score = similarity_scores[index][0]  # Extract similarity score from 2D array
    #     print(f"Job Title: {job_title}, Similarity Score: {similarity_score}")

    # Get indices of top 5 most similar job titles
    top_indices = np.argsort(similarity_scores, axis=0)[-5:][::-1].flatten()

    # Collect top 5 most similar job titles and their similarity scores in a list
    top_job_titles = []
    for index in top_indices:
        job_title = df['Job Title'][index]
        similarity_score = similarity_scores[index][0]  # Extract similarity score from 2D array
        top_job_titles.append((job_title, similarity_score))

    job_title, similarity = top_job_titles[0]


    #replace with the highest similarity (title normalization)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tech_df = pd.read_csv(r"Models\techh.csv")
    target_term = job_title
    job_titles = tech_df["job titles"]
    target_embedding = model.encode([target_term])
    job_title_embeddings = model.encode(job_titles)
    similarities = cosine_similarity(target_embedding, job_title_embeddings)[0]
    max_index = similarities.argmax()

    job_title = job_titles[max_index]

    return job_title, similarity 