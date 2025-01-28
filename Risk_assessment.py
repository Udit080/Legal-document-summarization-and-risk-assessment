import os
import re
import json
from transformers import pipeline

import pandas as pd


# Step 1: Load and preprocess the document

def load_and_preprocess(file_path, max_chunk_size=1500, min_chunk_size=700):
    

    # Load the document
    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()

    # Split the document dynamically based on the size of the text
    chunks = []
    start = 0
    while start < len(document):
        end = min(start + max_chunk_size, len(document))  # Ensure chunk doesn't exceed max size
        chunk = document[start:end]

        # If chunk is too large, split it further
        if len(chunk) > min_chunk_size:
            chunks.append(chunk)
        else:
            chunks[-1] += chunk  # Merge with previous chunk if it's small

        start = end

    return chunks

# Step 2: Risk Detection using huggingface

def analyze_text_for_risks_and_obligations(chunks):
    model_name = "google/flan-t5-base"  # Instruction-tuned model
    nlp = pipeline("text2text-generation", model=model_name)

    results = []

    for chunk in chunks:
        # Generate analysis for potential risks
        prompt_risks = (
            "Carefully analyze the following text for **potential risks**. "
            "Provide a thorough and detailed explanation of these risks, "
            "highlighting any concerns that may not be immediately obvious:\n\n" + chunk
        )
        risks_result = nlp(prompt_risks, max_length=512, do_sample=False)

        # Generate analysis for hidden obligations or dependencies
        prompt_obligations = (
            "Identify any **hidden obligations** or **dependencies** in the text. "
            "Be explicit about what these obligations may entail and explain their potential consequences:\n\n" + chunk
        )
        obligations_result = nlp(prompt_obligations, max_length=512, do_sample=False)

        # Generate actionable recommendations based on risks and obligations
        prompt_recommendations = (
            "Based on the **identified risks** and **hidden obligations** from the text, "
            "provide **specific and actionable recommendations** for addressing or mitigating these issues. "
            "Be practical, concise, and prioritize high-impact suggestions:\n\n" + chunk
        )
        recommendations_result = nlp(prompt_recommendations, max_length=512, do_sample=False)

        # Append results for each chunk
        results.append({
            "context": chunk,
            "risks_analysis": risks_result[0]['generated_text'],
            "obligations_analysis": obligations_result[0]['generated_text'],
            "recommendations": recommendations_result[0]['generated_text']
        })

    return results

# Step 3: Main execution function

def main(file_path):
    print("Loading and Preprocessing document...")
    chunks = load_and_preprocess(file_path)

    print("Analyzing & Detecting Risks and Generating Recommendations...")
    analysis = analyze_text_for_risks_and_obligations(chunks)

    # Save results to a JSON file
    output_path = "risk_analysis.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)

    print(f"Analysis complete. Results saved to {output_path}")
    with open("risk_analysis.json", "r", encoding="utf-8") as f:
     data = json.load(f)

# Print each entry's context and analysis components
    for entry in data:
     print("  # Context:\n")
     print("    ", entry["context"])
     print("\n" + "-" * 80)

     print("  # Risks Analysis:\n")
     print("    ", entry["risks_analysis"])
     print("\n" + "-" * 80)

     print("  # Obligations Analysis:\n")
     print("    ", entry["obligations_analysis"])
     print("\n" + "-" * 80)

     print("  # Recommendations:\n")
     print("    ", entry["recommendations"])
     print("=" * 300 + "\n")


    risks_df = pd.DataFrame(data)
     # print(df.head())
    return risks_df

# Execute the pipeline
if __name__ == "__main__":
    file_path = "C:\Certificates\Infosys\PROJECT\Will Doc.txt"
    main(file_path)
    
    


