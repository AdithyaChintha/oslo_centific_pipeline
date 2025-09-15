#!/usr/bin/env python3
"""
Domain Classification System using an Optimal RAG Technique:
The Reflective RAG Analyst.

This script:
1. Expands domain definitions into rich queries using an LLM.
2. Indexes all scene descriptions into a vector store.
3. For each domain, retrieves evidence using the expanded query.
4. Uses a Chain-of-Thought LLM prompt to analyze and score evidence.
5. Classifies based on the highest, most well-reasoned score.
"""

import os
import json
import glob
from typing import List, Dict, Any
from pathlib import Path
import logging
from datetime import datetime
import time

# LangChain imports
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Use the official LangChain integration for Groq
from langchain_groq import ChatGroq


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DomainClassifier:
    """
    Implements the Reflective RAG Analyst for accurate domain classification.
    """
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        # Base domains for expansion
        self.domains = {
            "Food & Mealtime": "Activities related to preparing, cooking, serving, eating meals.",
            "Personal Care & Hygiene": "Activities like grooming, bathing, dressing, and personal health routines.",
            "Household Movement": "General movement within the home, such as walking, sitting, or standing without a specific task.",
            "Cleaning & Maintenance": "Tasks like dusting, vacuuming, washing dishes, doing laundry, or home repairs.",
            "Work & Study": "Activities related to professional work, homework, reading, or using a computer for tasks.",
            "Leisure & Entertainment": "Watching TV, playing games, listening to music, hobbies, or relaxing.",
            "Exercise & Wellness": "Engaging in physical workouts, yoga, meditation, or other fitness activities.",
            "Social & Family Life": "Interacting with family members, talking, playing together, or hosting guests.",
            "Pet Care": "Activities related to feeding, playing with, or grooming pets.",
            "Home & Garden Projects": "Gardening, DIY projects, organizing, or redecorating the home.",
            "Shopping & Logistics": "Bringing in and unpacking groceries or packages, managing deliveries.",
            "Safety & Security": "Locking doors, checking security systems, or actions related to home safety."
        }
        
        # A fast model for scoring individual domains
        self.analyst_llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )
        # A more powerful model for the one-time query expansion task
        self.expansion_llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.2
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        self.vector_store = None
        self.expanded_domains = {}

    def _expand_domain_queries(self):
        """
        Uses a powerful LLM to expand each domain with related keywords and activities
        to create a high-quality, nuanced search query.
        """
        logger.info("🧠 Performing smart query expansion for all domains...")
        
        template = PromptTemplate(
            input_variables=["domain_name", "domain_description"],
            template="""
            You are a search query expert. For the domain '{domain_name}' described as '{domain_description}', generate a detailed, comprehensive search query.
            Include a list of specific actions, objects, synonyms, and related concepts.
            For example, for 'Cleaning & Maintenance', you should include 'washing dishes, wiping counters, mopping floors, tidying up, doing laundry'.
            
            Expanded Query:
            """
        )
        
        for name, description in self.domains.items():
            prompt = template.format(domain_name=name, domain_description=description)
            try:
                response = self.expansion_llm.invoke(prompt)
                self.expanded_domains[name] = response.content
                logger.info(f"  -> Expanded query for '{name}'")
            except Exception as e:
                logger.error(f"Failed to expand query for {name}: {e}")
                # Fallback to the basic description
                self.expanded_domains[name] = description
            time.sleep(1) # Pace the calls
        logger.info("✅ All domain queries expanded.")

    def collect_scene_descriptions(self, base_path: str = "./outtrymain") -> List[Document]:
        """Collects all scene descriptions and returns them as LangChain Documents."""
        logger.info(f"🔍 Collecting scene descriptions from {base_path}")
        documents = []
        pattern = os.path.join(base_path, "**/scene_output/*_scene_detection_results.json")
        scene_files = glob.glob(pattern, recursive=True)
        
        for file_path in scene_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if "scenes" in data:
                    for scene in data["scenes"]:
                        content = f"Scene from {scene['start_time']:.1f}s to {scene['end_time']:.1f}s: {scene.get('description', '')}"
                        doc = Document(page_content=content, metadata={"file_path": file_path, "start": scene['start_time']})
                        documents.append(doc)
            except Exception as e:
                logger.error(f"❌ Error reading {file_path}: {e}")
        
        logger.info(f"✅ Collected {len(documents)} total scenes as documents")
        return documents

    def _build_vector_store(self, documents: List[Document]):
        """Builds a FAISS vector store from the collected documents."""
        logger.info(f"🛠️ Building vector store from {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("✅ Vector store built successfully.")

    def _get_domain_evidence(self, domain_name: str, k: int = 5) -> List[Document]:
        """Retrieves top-k relevant documents using the expanded query."""
        if not self.vector_store: return []
        expanded_query = self.expanded_domains.get(domain_name, self.domains[domain_name])
        return self.vector_store.similarity_search(expanded_query, k=k)

    def _score_domain_evidence(self, domain_name: str, evidence_docs: List[Document]) -> Dict[str, Any]:
        """Uses a Chain-of-Thought prompt to deeply analyze and score evidence."""
        if not evidence_docs:
            return {"score": 0, "reasoning": "No relevant scenes found."}
            
        evidence_text = "\n".join([f"- {doc.page_content}" for doc in evidence_docs])
        
        # This is the advanced, Chain-of-Thought prompt
        prompt_template = PromptTemplate(
            input_variables=["domain_name", "domain_description", "evidence_text"],
            template="""
            You are an expert analyst. Your task is to score how well a set of scenes fit the domain '{domain_name}'.
            Domain Description: {domain_description}

            **CRITICAL RULE:** Do not confuse related activities. For example, **washing dishes** is a 'Cleaning & Maintenance' task, NOT a 'Food & Mealtime' task, even though it involves plates. Focus on the PRIMARY ACTION.

            **Retrieved Scenes (Evidence):**
            {evidence_text}

            **Your Chain of Thought (Analyze Step-by-Step):**
            1.  **Primary Action**: What is the main action or activity described in the evidence?
            2.  **Comparison**: Does this primary action directly match the domain description of '{domain_name}'?
            3.  **Reasoning**: Based on the comparison and the critical rule, briefly explain your conclusion.
            4.  **Score**: Assign a confidence score from 0 (no match) to 10 (perfect match).

            Respond ONLY with a valid JSON object in the following format:
            {{"score": <score_integer>, "reasoning": "<your_brief_reasoning>"}}
            """
        )
        
        prompt = prompt_template.format(
            domain_name=domain_name,
            domain_description=self.domains[domain_name],
            evidence_text=evidence_text
        )
        
        try:
            response = self.analyst_llm.invoke(prompt)
            result = json.loads(response.content)
            result['score'] = int(result.get('score', 0))
            return result
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM response for domain '{domain_name}': {e}")
            return {"score": 0, "reasoning": "Error during analysis."}

    def process_and_classify(self, base_path: str = "./outtrymain") -> Dict[str, Any]:
        """Runs the full Reflective RAG pipeline."""
        logger.info("🚀 Starting Reflective RAG classification process...")
        start_time = datetime.now()
        
        # 0. Expand Queries
        self._expand_domain_queries()

        # 1. Index
        documents = self.collect_scene_descriptions(base_path)
        if not documents: return {"error": "No scene data found."}
        self._build_vector_store(documents)
        
        domain_scores = []
        
        # 2. Retrieve, Analyze & Score for each domain
        for name in self.domains.keys():
            logger.info(f"🔄 Analyzing domain: {name}")
            evidence = self._get_domain_evidence(name, k=5)
            score_result = self._score_domain_evidence(name, evidence)
            
            domain_scores.append({
                "domain": name,
                "score": score_result["score"],
                "reasoning": score_result["reasoning"],
                "evidence": [doc.page_content for doc in evidence]
            })
            time.sleep(1) # Pace API calls

        # 3. Classify
        sorted_scores = sorted(domain_scores, key=lambda x: x['score'], reverse=True)
        best_match = sorted_scores[0]
        
        end_time = datetime.now()
        
        results = {
            "processing_time_seconds": (end_time - start_time).total_seconds(),
            "total_scenes_indexed": len(documents),
            "best_match": best_match,
            "all_domain_scores": sorted_scores
        }
        
        logger.info(f"✅ Classification complete. Best match: '{best_match['domain']}' (Score: {best_match['score']})")
        return results

    def save_results(self, results: Dict[str, Any], output_file: str = "./outtrymain/domain_classification_rag_results.json"):
        """Saves the classification results to a JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {output_file}")


def main():
    """Main function to run the domain classification."""
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
    if not GROQ_API_KEY or "your_groq_api_key_here" in GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set. Please set it as an environment variable or in the script.")
        return

    classifier = DomainClassifier(GROQ_API_KEY)
    results = classifier.process_and_classify("./outtrymain")
    
    if "error" in results:
        print(f"\n❌ CLASSIFICATION FAILED: {results['error']}")
        return
        
    classifier.save_results(results)
    
    best_match = results['best_match']
    
    print("\n" + "="*60)
    print("🎯 REFLECTIVE RAG DOMAIN CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Processing Time: {results['processing_time_seconds']:.2f} seconds")
    print(f"Total Scenes Indexed: {results['total_scenes_indexed']}")
    print("-" * 60)
    print(f"🏆 Best Matched Domain: {best_match['domain']}")
    print(f"Confidence Score: {best_match['score']}/10")
    print(f"Reasoning: {best_match['reasoning']}")
    print("-" * 60)
    print("Top Evidence Found:")
    for i, evidence in enumerate(best_match['evidence'][:3], 1):
        print(f"  {i}. {evidence}")
    print("="*60)


if __name__ == "__main__":
    main()