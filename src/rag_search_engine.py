import json
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from config import PINECONE_API_KEY
import re
from collections import defaultdict

class RAGSearchEngine:
    """
    Multi-stage RAG Search Engine:
    1. Filter: Lọc 1000 sản phẩm dựa trên filter criteria
    2. TF-IDF: Lọc xuống 100 sản phẩm tốt nhất
    3. Semantic Search: Tìm 10 vector tương đồng nhất
    4. Rerank: Sắp xếp lại để lấy 5 kết quả tốt nhất cho LLM
    """
    
    def __init__(self, 
                 model_name: str = "bkai-foundation-models/vietnamese-bi-encoder",
                 pinecone_index_name: str = "product-embeddings"):
        """
        Initialize RAG Search Engine
        
        Args:
            model_name: Sentence transformer model name
            pinecone_index_name: Pinecone index name
        """
        print("Initializing RAG Search Engine...")
        
        # Load sentence transformer model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize Pinecone
        print("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(pinecone_index_name)
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # Vietnamese stop words can be added
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Load product data
        self.products_data = self._load_products_data()
        self.chunked_data = self._load_chunked_data()
        
        # Prepare TF-IDF corpus
        self._prepare_tfidf_corpus()
        
        print("RAG Search Engine initialized successfully!")
    
    def _load_products_data(self) -> List[Dict[str, Any]]:
        """Load original products data"""
        try:
            with open('data/products_info.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: products_info.json not found")
            return []
    
    def _load_chunked_data(self) -> List[Dict[str, Any]]:
        """Load chunked documents data"""
        try:
            with open('data/chunked_documents.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: chunked_documents.json not found")
            return []
    
    def _prepare_tfidf_corpus(self):
        """Prepare TF-IDF corpus from product data"""
        print("Preparing TF-IDF corpus...")
        
        # Create corpus for TF-IDF
        self.tfidf_corpus = []
        self.product_id_to_index = {}
        
        for idx, product in enumerate(self.products_data):
            # Combine all text fields for TF-IDF
            text_fields = [
                product.get('name', ''),
                product.get('english_name', ''),
                product.get('brand', ''),
                product.get('categorys', ''),
                product.get('category_name', ''),
                product.get('descriptioninfo', ''),
                product.get('specificationinfo', ''),
                product.get('ingredientinfo', ''),
                product.get('guideinfo', ''),
            ]
            
            # Add comments
            comments = product.get('comments', [])
            if comments:
                text_fields.extend(comments)
            
            # Combine all text
            combined_text = ' '.join([str(field) for field in text_fields if field])
            combined_text = self._clean_text(combined_text)
            
            self.tfidf_corpus.append(combined_text)
            self.product_id_to_index[product.get('data_product')] = idx
        
        # Fit TF-IDF vectorizer
        if self.tfidf_corpus:
            print("Fitting TF-IDF vectorizer...")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.tfidf_corpus)
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Vietnamese
        text = re.sub(r'[^\w\sÀ-ỹ]', ' ', text)
        
        return text.strip().lower()
    
    def stage1_filter_products(self, 
                              query: str,
                              filters: Optional[Dict[str, Any]] = None,
                              limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Stage 1: Filter products based on criteria
        
        Args:
            query: Search query
            filters: Filter criteria (brand, category, price_range, etc.)
            limit: Maximum number of products to return
            
        Returns:
            List of filtered products
        """
        print(f"Stage 1: Filtering products (target: {limit})")
        
        filtered_products = []
        
        for product in self.products_data:
            # Apply filters if provided
            if filters:
                if not self._apply_filters(product, filters):
                    continue
            
            # Basic text matching for initial filtering
            if self._basic_text_match(product, query):
                filtered_products.append(product)
                
                if len(filtered_products) >= limit:
                    break
        
        print(f"Stage 1 completed: {len(filtered_products)} products filtered")
        return filtered_products
    
    def _apply_filters(self, product: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filter criteria to product"""
        
        # Brand filter
        if 'brands' in filters and filters['brands']:
            if product.get('brand', '').lower() not in [b.lower() for b in filters['brands']]:
                return False
        
        # Category filter
        if 'categories' in filters and filters['categories']:
            product_categories = [
                product.get('categorys', '').lower(),
                product.get('category_name', '').lower()
            ]
            if not any(cat in product_categories for cat in [c.lower() for c in filters['categories']]):
                return False
        
        # Price range filter
        if 'price_range' in filters and filters['price_range']:
            price = product.get('price', 0)
            if isinstance(price, (int, float)):
                min_price = filters['price_range'].get('min', 0)
                max_price = filters['price_range'].get('max', float('inf'))
                if not (min_price <= price <= max_price):
                    return False
        
        # Rating filter
        if 'min_rating' in filters and filters['min_rating']:
            rating = product.get('average_rating', 0)
            if isinstance(rating, (int, float)) and rating < filters['min_rating']:
                return False
        
        return True
    
    def _basic_text_match(self, product: Dict[str, Any], query: str) -> bool:
        """Basic text matching for initial filtering"""
        query_lower = query.lower()
        
        # Check in main fields
        searchable_fields = [
            product.get('name', ''),
            product.get('english_name', ''),
            product.get('brand', ''),
            product.get('categorys', ''),
            product.get('category_name', ''),
            product.get('descriptioninfo', ''),
        ]
        
        for field in searchable_fields:
            if query_lower in str(field).lower():
                return True
        
        return False
    
    def stage2_tfidf_ranking(self, 
                           filtered_products: List[Dict[str, Any]], 
                           query: str, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Stage 2: Use TF-IDF to rank and select top products
        
        Args:
            filtered_products: Products from stage 1
            query: Search query
            limit: Number of top products to return
            
        Returns:
            Top ranked products based on TF-IDF similarity
        """
        print(f"Stage 2: TF-IDF ranking (target: {limit})")
        
        if not filtered_products:
            return []
        
        # Get indices of filtered products
        filtered_indices = []
        for product in filtered_products:
            product_id = product.get('data_product')
            if product_id in self.product_id_to_index:
                filtered_indices.append(self.product_id_to_index[product_id])
        
        if not filtered_indices:
            return filtered_products[:limit]
        
        # Transform query using fitted TF-IDF vectorizer
        query_cleaned = self._clean_text(query)
        query_vector = self.tfidf_vectorizer.transform([query_cleaned])
        
        # Calculate similarities only for filtered products
        filtered_tfidf_matrix = self.tfidf_matrix[filtered_indices]
        similarities = cosine_similarity(query_vector, filtered_tfidf_matrix).flatten()
        
        # Sort by similarity and get top products
        sorted_indices = np.argsort(similarities)[::-1][:limit]
        
        top_products = []
        for idx in sorted_indices:
            original_idx = filtered_indices[idx]
            product = self.products_data[original_idx]
            product['tfidf_score'] = float(similarities[idx])
            top_products.append(product)
        
        print(f"Stage 2 completed: {len(top_products)} products ranked by TF-IDF")
        return top_products
    
    def stage3_semantic_search(self, 
                             tfidf_products: List[Dict[str, Any]], 
                             query: str, 
                             limit: int = 10) -> List[Dict[str, Any]]:
        """
        Stage 3: Semantic search using embeddings
        
        Args:
            tfidf_products: Products from stage 2
            query: Search query
            limit: Number of top semantic matches to return
            
        Returns:
            Top semantic matches
        """
        print(f"Stage 3: Semantic search (target: {limit})")
        
        if not tfidf_products:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query]).tolist()[0]
        
        # Get product IDs for filtering
        product_ids = [p.get('data_product') for p in tfidf_products]
        
        # Search in Pinecone with metadata filtering
        search_results = self.index.query(
            vector=query_embedding,
            top_k=limit * 5,  # Get more results to ensure we have enough after filtering
            include_metadata=True,
            filter={
                "product_id": {"$in": product_ids}
            }
        )
        
        # Group results by product_id and get best score for each product
        product_scores = defaultdict(list)
        for match in search_results['matches']:
            product_id = match['metadata']['product_id']
            product_scores[product_id].append({
                'score': match['score'],
                'metadata': match['metadata']
            })
        
        # Get best score for each product and sort
        best_matches = []
        for product_id, matches in product_scores.items():
            best_match = max(matches, key=lambda x: x['score'])
            best_matches.append({
                'product_id': product_id,
                'semantic_score': best_match['score'],
                'metadata': best_match['metadata']
            })
        
        # Sort by semantic score and limit
        best_matches.sort(key=lambda x: x['semantic_score'], reverse=True)
        best_matches = best_matches[:limit]
        
        # Merge with original product data
        semantic_products = []
        for match in best_matches:
            # Find original product data
            original_product = next(
                (p for p in tfidf_products if p.get('data_product') == match['product_id']), 
                None
            )
            if original_product:
                original_product['semantic_score'] = match['semantic_score']
                original_product['relevant_chunk'] = match['metadata'].get('text', '')
                semantic_products.append(original_product)
        
        print(f"Stage 3 completed: {len(semantic_products)} semantic matches")
        return semantic_products
    
    def stage4_rerank(self, 
                     semantic_products: List[Dict[str, Any]], 
                     query: str, 
                     limit: int = 5) -> List[Dict[str, Any]]:
        """
        Stage 4: Final reranking for best results
        
        Args:
            semantic_products: Products from stage 3
            query: Search query
            limit: Final number of products for LLM
            
        Returns:
            Final reranked products for LLM
        """
        print(f"Stage 4: Final reranking (target: {limit})")
        
        if not semantic_products:
            return []
        
        # Calculate combined score
        for product in semantic_products:
            tfidf_score = product.get('tfidf_score', 0)
            semantic_score = product.get('semantic_score', 0)
            
            # Weighted combination (can be tuned)
            combined_score = (
                0.3 * tfidf_score +  # TF-IDF weight
                0.7 * semantic_score  # Semantic weight
            )
            
            # Add bonus for high ratings and popularity
            rating_bonus = (product.get('average_rating', 0) / 5.0) * 0.1
            popularity_bonus = min(product.get('item_count_by', 0) / 1000, 1.0) * 0.1
            
            product['final_score'] = combined_score + rating_bonus + popularity_bonus
        
        # Sort by final score
        reranked_products = sorted(
            semantic_products, 
            key=lambda x: x['final_score'], 
            reverse=True
        )[:limit]
        
        print(f"Stage 4 completed: {len(reranked_products)} final products for LLM")
        return reranked_products
    
    def search(self, 
               query: str, 
               filters: Optional[Dict[str, Any]] = None,
               stage1_limit: int = 1000,
               stage2_limit: int = 100,
               stage3_limit: int = 10,
               final_limit: int = 5) -> Dict[str, Any]:
        """
        Complete RAG search pipeline
        
        Args:
            query: Search query
            filters: Filter criteria
            stage1_limit: Products after filtering
            stage2_limit: Products after TF-IDF
            stage3_limit: Products after semantic search
            final_limit: Final products for LLM
            
        Returns:
            Search results with metadata
        """
        print(f"\n🔍 Starting RAG Search Pipeline for query: '{query}'")
        print("=" * 60)
        
        # Stage 1: Filter
        stage1_results = self.stage1_filter_products(query, filters, stage1_limit)
        
        # Stage 2: TF-IDF
        stage2_results = self.stage2_tfidf_ranking(stage1_results, query, stage2_limit)
        
        # Stage 3: Semantic Search
        stage3_results = self.stage3_semantic_search(stage2_results, query, stage3_limit)
        
        # Stage 4: Rerank
        final_results = self.stage4_rerank(stage3_results, query, final_limit)
        
        print("=" * 60)
        print("🎯 RAG Search Pipeline completed!")
        
        return {
            'query': query,
            'filters': filters,
            'results': final_results,
            'pipeline_stats': {
                'stage1_count': len(stage1_results),
                'stage2_count': len(stage2_results),
                'stage3_count': len(stage3_results),
                'final_count': len(final_results)
            }
        }
    
    def format_for_llm(self, search_results: Dict[str, Any]) -> str:
        """
        Format search results for LLM input
        
        Args:
            search_results: Results from search method
            
        Returns:
            Formatted string for LLM
        """
        products = search_results['results']
        
        if not products:
            return "Không tìm thấy sản phẩm phù hợp với yêu cầu."
        
        formatted_text = f"Dựa trên truy vấn '{search_results['query']}', đây là {len(products)} sản phẩm phù hợp nhất:\n\n"
        
        for i, product in enumerate(products, 1):
            formatted_text += f"**Sản phẩm {i}:**\n"
            formatted_text += f"- Tên: {product.get('name', 'N/A')}\n"
            formatted_text += f"- Thương hiệu: {product.get('brand', 'N/A')}\n"
            formatted_text += f"- Giá: {product.get('price', 'N/A'):,} VNĐ\n"
            formatted_text += f"- Đánh giá: {product.get('average_rating', 'N/A')}/5 ({product.get('total_rating', 0)} đánh giá)\n"
            formatted_text += f"- Danh mục: {product.get('category_name', 'N/A')}\n"
            
            # Add relevant chunk if available
            if product.get('relevant_chunk'):
                formatted_text += f"- Thông tin liên quan: {product['relevant_chunk'][:200]}...\n"
            
            formatted_text += f"- Độ phù hợp: {product.get('final_score', 0):.3f}\n\n"
        
        return formatted_text


# Example usage and testing
if __name__ == "__main__":
    # Initialize search engine
    search_engine = RAGSearchEngine()
    
    # Example search
    query = "sữa rửa mặt cho da dầu"
    filters = {
        'categories': ['skincare', 'cleanser'],
        'price_range': {'min': 0, 'max': 500000},
        'min_rating': 4.0
    }
    
    # Perform search
    results = search_engine.search(
        query=query,
        filters=filters,
        stage1_limit=1000,
        stage2_limit=100,
        stage3_limit=10,
        final_limit=5
    )
    
    # Format for LLM
    llm_input = search_engine.format_for_llm(results)
    print("\n" + "="*60)
    print("📝 FORMATTED OUTPUT FOR LLM:")
    print("="*60)
    print(llm_input)