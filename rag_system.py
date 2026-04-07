"""
RAG (Retrieval-Augmented Generation) System for WRC Resources

This module implements a semantic search and question-answering system
using embeddings and vector similarity search.
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import json
from dotenv import load_dotenv

# Optional: OpenAI for embeddings and chat
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGSystem:
    """
    Retrieval-Augmented Generation system for resource search and Q&A.
    
    Supports both local embeddings (sentence-transformers) and OpenAI embeddings.
    """
    
    def __init__(self, db_path: str = "wrc_resources.db", 
                 use_local_embeddings: bool = True):
        load_dotenv()
        
        self.db_path = db_path
        self.use_local_embeddings = use_local_embeddings
        
        # Initialize embedding model
        if use_local_embeddings:
            print("Loading local embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
            self.model_name = 'all-MiniLM-L6-v2'
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install openai")
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.embedding_dim = 1536  # OpenAI ada-002
            self.model_name = 'text-embedding-ada-002'
            
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self.use_local_embeddings:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return np.array(response.data[0].embedding)
            
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        if self.use_local_embeddings:
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        else:
            embeddings = []
            # Process in batches to avoid API limits
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            return np.array(embeddings)
            
    def create_embeddings_for_all_resources(self):
        """Create embeddings for all resources in the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Clear existing embeddings to avoid duplicates
        print("Clearing old embeddings...")
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        
        # Get all resources
        cursor.execute("""
            SELECT 
                r.resource_id,
                r.organization_name,
                r.description,
                r.services,
                r.resource_type,
                i.ocr_text,
                GROUP_CONCAT(rc.category, ', ') as categories
            FROM resources r
            JOIN images i ON r.image_id = i.image_id
            LEFT JOIN resource_categories rc ON r.resource_id = rc.resource_id
            GROUP BY r.resource_id
        """)
        
        resources = cursor.fetchall()
        print(f"Creating embeddings for {len(resources)} resources...")
        
        # Prepare texts for embedding
        texts = []
        resource_ids = []
        
        for resource in resources:
            # Combine relevant fields into a single text
            parts = []
            
            if resource['organization_name']:
                parts.append(f"Organization: {resource['organization_name']}")
            if resource['resource_type']:
                parts.append(f"Type: {resource['resource_type']}")
            if resource['categories']:
                parts.append(f"Categories: {resource['categories']}")
            if resource['description']:
                parts.append(f"Description: {resource['description']}")
            if resource['services']:
                parts.append(f"Services: {resource['services']}")
                
            # Add a portion of OCR text
            if resource['ocr_text']:
                ocr_snippet = resource['ocr_text'][:1000]
                parts.append(f"Full text: {ocr_snippet}")
                
            combined_text = "\n".join(parts)
            texts.append(combined_text)
            resource_ids.append(resource['resource_id'])
            
        # Generate embeddings
        embeddings = self.get_embeddings_batch(texts)
        
        # Store embeddings in database
        print("Storing embeddings in database...")
        for resource_id, text, embedding in zip(resource_ids, texts, embeddings):
            embedding_json = json.dumps(embedding.tolist())
            cursor.execute("""
                INSERT INTO embeddings (resource_id, embedding_text, 
                                      embedding_vector, embedding_model)
                VALUES (?, ?, ?, ?)
            """, (resource_id, text, embedding_json, self.model_name))
            
        conn.commit()
        conn.close()
        print(f"Successfully created and stored {len(embeddings)} embeddings")
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def search(self, query: str, top_k: int = 5, 
               current_only: bool = True) -> List[Dict]:
        """
        Semantic search for resources using embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            current_only: Only return current (non-outdated) resources
            
        Returns:
            List of resource dicts with similarity scores
        """
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        
        # Get all embeddings from database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query_sql = """
            SELECT 
                e.embedding_id,
                e.resource_id,
                e.embedding_text,
                e.embedding_vector,
                r.organization_name,
                r.resource_type,
                r.description,
                r.address,
                r.phone,
                r.email,
                r.website,
                r.hours,
                r.eligibility,
                r.services,
                r.is_current,
                r.currency_score,
                i.file_path,
                i.binder_name,
                i.ocr_text,
                i.original_filename
            FROM embeddings e
            JOIN resources r ON e.resource_id = r.resource_id
            JOIN images i ON r.image_id = i.image_id
        """
        
        if current_only:
            query_sql += " WHERE r.is_current = 1"
            
        cursor.execute(query_sql)
        
        results = []
        for row in cursor.fetchall():
            # Parse embedding vector
            stored_embedding = np.array(json.loads(row['embedding_vector']))
            
            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, stored_embedding)
            
            results.append({
                'resource_id': row['resource_id'],
                'organization_name': row['organization_name'],
                'resource_type': row['resource_type'],
                'description': row['description'],
                'address': row['address'],
                'phone': row['phone'],
                'email': row['email'],
                'website': row['website'],
                'hours': row['hours'],
                'eligibility': row['eligibility'],
                'services': row['services'],
                'is_current': bool(row['is_current']),
                'currency_score': row['currency_score'],
                'file_path': row['file_path'],
                'binder_name': row['binder_name'],
                'ocr_text': row['ocr_text'],
                'similarity_score': float(similarity),
            })
            
        conn.close()
        
        # CCSF official reporting/response channels - highest priority for crisis
        CCSF_REPORTING_RESOURCES = [
            'Title IX Office',
            'CCSF District Police',
        ]
        
        # Critical crisis resources with 24-hour hotlines - immediate crisis response
        CRITICAL_CRISIS_RESOURCES = [
            'San Francisco Women Against Rape (SFWAR)',
        ]
        
        # Mark and boost resources by priority tier
        for result in results:
            org_name = result.get('organization_name') or ''
            is_ccsf = result.get('binder_name') == 'CCSF_Website' or '(CCSF)' in org_name
            
            # Check if this is a CCSF official reporting channel
            is_ccsf_reporting = is_ccsf and any(r in org_name for r in CCSF_REPORTING_RESOURCES)
            is_crisis_hotline = org_name in CRITICAL_CRISIS_RESOURCES
            
            if is_ccsf_reporting:
                result['priority_tier'] = 0  # Tier 0: CCSF official reporting
                result['similarity_score'] = result['similarity_score'] + 0.15
            elif is_crisis_hotline:
                result['priority_tier'] = 1  # Tier 1: 24-hour crisis hotlines
                result['similarity_score'] = result['similarity_score'] + 0.14
            elif is_ccsf:
                result['priority_tier'] = 2  # Tier 2: Other CCSF resources (education, peer support)
                result['similarity_score'] = result['similarity_score'] + 0.13
            else:
                result['priority_tier'] = 3  # Tier 3: All other resources
            
            result['is_ccsf_resource'] = is_ccsf
            result['is_crisis_resource'] = is_crisis_hotline
        
        # Sort by boosted similarity score
        results.sort(key=lambda x: -x['similarity_score'])
        
        # Get top K results
        top_results = results[:top_k]
        
        # Re-sort with priority tiers:
        # 0. CCSF official reporting (Title IX, Police)
        # 1. Critical 24-hour crisis hotlines (SFWAR)
        # 2. Other CCSF resources (education, peer support)
        # 3. Everything else
        # Within each tier, sort by similarity score
        top_results.sort(key=lambda x: (x.get('priority_tier', 3), -x['similarity_score']))
        
        return top_results
        
    def ask_question(self, question: str, top_k: int = 3) -> str:
        """
        Answer a question using RAG approach.
        
        Retrieves relevant resources and generates an answer using GPT.
        Requires OpenAI API key.
        """
        if not OPENAI_AVAILABLE:
            return "OpenAI not available. Please install: pip install openai"
            
        if self.use_local_embeddings:
            # Initialize OpenAI client for chat completion
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            client = self.client
            
        # Search for relevant resources
        relevant_resources = self.search(question, top_k=top_k)
        
        if not relevant_resources:
            return "I couldn't find any relevant resources for your question."
            
        # Format context from retrieved resources
        context_parts = []
        for i, resource in enumerate(relevant_resources, 1):
            context = f"\n--- Resource {i} ---\n"
            if resource['organization_name']:
                context += f"Organization: {resource['organization_name']}\n"
            if resource['resource_type']:
                context += f"Type: {resource['resource_type']}\n"
            if resource['description']:
                context += f"Description: {resource['description']}\n"
            if resource['address']:
                context += f"Address: {resource['address']}\n"
            if resource['phone']:
                context += f"Phone: {resource['phone']}\n"
            if resource['email']:
                context += f"Email: {resource['email']}\n"
            if resource['website']:
                context += f"Website: {resource['website']}\n"
            if resource['hours']:
                context += f"Hours: {resource['hours']}\n"
            if resource['eligibility']:
                context += f"Eligibility: {resource['eligibility']}\n"
                
            context_parts.append(context)
            
        full_context = "\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are a helpful assistant for the Women's Resource Center at City College of San Francisco. 
Your role is to help people find resources and services. Use the provided resource information to answer questions 
accurately and helpfully. If you're not sure about something, say so. Always prioritize the safety and well-being 
of the people you're helping."""

        user_prompt = f"""Based on the following resources from our database, please provide resources relevant to a student based on the following:

Question: {question}

Available Resources:
{full_context}

Please provide a helpful, accurate answer based on these resources. Include specific contact information when relevant."""

        # Get completion from GPT
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"


def main():
    """Test the RAG system."""
    print("Initializing RAG system...")
    rag = RAGSystem(use_local_embeddings=True)
    
    # Test search
    query = "housing assistance for women"
    print(f"\nSearching for: '{query}'")
    results = rag.search(query, top_k=3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['organization_name']}")
        print(f"   Type: {result['resource_type']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Current: {'Yes' if result['is_current'] else 'No'}")


if __name__ == "__main__":
    main()

