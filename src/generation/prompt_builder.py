class PromptBuilder:

    def build(self, query, documents):

        context = ""

        for i, doc in enumerate(documents, start=1):

            context += f"""
Document {i}

Title:
{doc.title}

Body:
{doc.body}

-------------------------

"""

        prompt = f"""
You are an expert software engineer.

Answer the question ONLY using the context.

If the answer is not in the context,
say you don't know.

Context

{context}

Question

{query}

Answer
[Answer will be generated here later]
"""

        return prompt
    


