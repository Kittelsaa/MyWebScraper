from langchain_ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def parse_with_ollama(dom_chunks, parse_description):
    try:
        llm = Ollama(model="llama2", temperature=0)
        prompt = PromptTemplate.from_template(
            """
            You are a precise web content parser tasked with extracting specific information.
            CONTENT TO ANALYZE:
            {dom_content}
            EXTRACTION REQUEST:
            {parse_description}
            INSTRUCTIONS:
            1. Focus only on extracting information that exactly matches the extraction request
            2. Return only the extracted data, no explanations or additional text
            3. If no matching information is found, return an empty string ('')
            4. Format the output in a clean, readable way
            5. Do not include any commentary, headers, or notes
            6. Do not make assumptions about missing data
            EXTRACTED INFORMATION:
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        results = []
        for chunk in dom_chunks:
            try:
                response = chain.invoke(
                    {"dom_content": chunk, "parse_description": parse_description}
                )
                results.append(response)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue   
        return "\n".join(results)
        
    except ConnectionError:
        return "Error: Cannot connect to Ollama server. Please make sure Ollama is running (use 'ollama serve' in terminal)"
    except Exception as e:
        return f"Error: {str(e)}"