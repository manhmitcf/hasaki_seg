import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class TextChunker:
    """
    Utility class for chunking long text into smaller, semantically meaningful pieces
    """
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
        return text
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using Vietnamese sentence patterns
        """
        # Vietnamese sentence endings
        sentence_endings = r'[.!?](?:\s|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or len(text) <= self.chunk_size:
            return [{"text": self.clean_text(text), "chunk_id": 0}]
        
        text = self.clean_text(text)
        chunks = []
        
        # Try to split by sentences first for better semantic coherence
        sentences = self.split_by_sentences(text)
        
        if len(sentences) > 1:
            # Group sentences into chunks
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                # If adding this sentence would exceed chunk size, save current chunk
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_id": chunk_id
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence

                    chunk_id += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": chunk_id,
                })
        else:
            # Fallback to character-based chunking
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk_text = text[i:i + self.chunk_size]
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": i // (self.chunk_size - self.overlap),
                })
        
        return chunks
    def general_information_embedding(self, product: Dict[str, Any], model: SentenceTransformer) -> List[float]:
        """
        Extract general information from a product dictionary
        
        Args:
            product: Product dictionary containing various fields
            
        Returns:
            Sentence summarizing key product information
        """
        name = "Tên sản phẩm: " + product.get('name', 'Không có tên')
        english_name = "Tên tiếng anh: " + product.get('english_name', 'Không có tên tiếng anh')
        brand = "Thương hiệu: " + product.get('brand', 'Không có thương hiệu')
        category_name = "Danh mục con: " + product.get('category_name', 'Không có danh mục con')
        price = "Giá hiện tại: " + str(product.get('price', 'Không có giá'))
        market_price = "Giá thị trường: " + str(product.get('market_price', 'Không có giá thị trường'))
        total_rating = "Tổng số đánh giá: " + str(product.get('total_rating', 'Không có đánh giá'))
        average_rating = "Đánh giá trung bình: " + str(product.get('average_rating', 'Không có đánh giá'))
        comment_count = "Số lượng bình luận: " + str(product.get('comment', 'Không có bình luận'))
        categorys = "Danh mục chính: " + product.get('categorys', 'Không có danh mục chính')
        item_count_by = "Số lượng sản phẩm đã mua: " + str(product.get('item_count_by', 'Không có số lượng sản phẩm đã mua'))
        data_variant = "Dữ liệu biến thể: " + str(product.get('data_variant', 'Không có dữ liệu biến thể'))
        stars = "Chi tiết đánh giá: " + self.clean_text(str(product.get('stars', 'Không có chi tiết đánh giá')))
        
        # Combine all information into a single string
        general_info = f"{name}. {english_name}. {brand}. {category_name}. {price}. {market_price}. {total_rating}. {average_rating}. {comment_count}. {categorys}. {item_count_by}. {data_variant}. {stars}."
        
        #Embedding the general information
        general_info = self.clean_text(general_info)
        general_embedding = model.encode([general_info]).tolist()[0]
        return general_embedding, general_info
    def comment_embedding(self, product: Dict[str, Any], model: SentenceTransformer) -> List[Dict[str, Any]]:
        """
        Extract comment information from a product dictionary
        Comments : List

        Args:
            product: Product dictionary containing various fields

        Returns:
            Sentence summarizing key product information
        """
        comments = product.get('comments', [])
        if len(comments) ==0:
            return True
        
        comments_embedding = []
        comment_id = 0
        for comment in comments:
            comment = comment.strip()
            if comment:
                comment_info = f"Hỏi đáp của khách hành và Shop: {comment}"
                comment_embedding = self.clean_text(comment_info)
                comment_embedding = model.encode([comment_info]).tolist()[0]

                comments_embedding.append({
                    "comment_id": comment_id,
                    "comment": comment,
                    "vector": comment_embedding
                })
                comment_id += 1
           
        return comments_embedding
    def specificationinfo_embedding(self, product: Dict[str, Any], model: SentenceTransformer) -> List[float]:
        """
        Extract specification information from a product dictionary
        
        Args:
            product: Product dictionary containing specification info
            
        Returns:
            Sentence summarizing key specification information
        """
        specification = product.get('specificationinfo', '')
        if not specification:
            return True
        
        # Clean and encode the specification text
        cleaned_specification = self.clean_text(specification)




        spec_embedding = model.encode([cleaned_specification]).tolist()[0]
        
        return spec_embedding
    def chunk_product_ingredientinfo(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract ingredient information from a product dictionary
        
        Args:
            product: Product dictionary containing ingredient info
            
        Returns:
            Sentence summarizing key ingredient information
        """
        ingredient = product.get('ingredientinfo', '')
        if not ingredient:
            return True
        
        # Clean and encode the ingredient text
        cleaned_ingredient = self.clean_text(ingredient)
        
        chunks = self.chunk_text(cleaned_ingredient)
        chunked_products = []
        
        for chunk in chunks:
            # Create a new document for each chunk
            chunk_doc = {
                # Core product info
                'product_id': product.get('data_product'),
                'name': product.get('name'),
                # Chunk-specific info
                'chunk_text': chunk['text'],
                'chunk_id': chunk['chunk_id'],
            }
            
            chunked_products.append(chunk_doc)
        
        return chunked_products
    

    def chunk_product_description(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks for a product's description with metadata
        
        Args:
            product: Product dictionary containing description info
            
        Returns:
            List of chunk documents ready for vector storage
        """
        description = product.get('descriptioninfo', '')
        if not description:
            return []
        
        chunks = self.chunk_text(description)
        chunked_products = []
        
        for chunk in chunks:
            # Create a new document for each chunk
            chunk_doc = {
                # Core product info
                'product_id': product.get('data_product'),
                'name': product.get('name'),
                # Chunk-specific info
                'chunk_text': chunk['text'],
                'chunk_id': chunk['chunk_id'],
            }
            
            chunked_products.append(chunk_doc)
        
        return chunked_products
    def chuck_product_guideinfo(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract guide information from a product dictionary
        
        Args:
            product: Product dictionary containing guide info
            
        Returns:
            Sentence summarizing key guide information
        """
        guide = product.get('guideinfo', '')
        if not guide:
            return True
        
        # Clean and encode the guide text
        cleaned_guide = self.clean_text(guide)
        
        chunks = self.chunk_text(cleaned_guide)
        chunked_products = []
        
        for chunk in chunks:
            # Create a new document for each chunk
            chunk_doc = {
                # Core product info
                'product_id': product.get('data_product'),
                'name': product.get('name'),
                # Chunk-specific info
                'chunk_text': chunk['text'],
                'chunk_id': chunk['chunk_id'],
            }
            
            chunked_products.append(chunk_doc)
        
        return chunked_products
def process_products_with_chunking(products: List[Dict[str, Any]], 
                                 model: SentenceTransformer,
                                 chunk_size: int = 300, 
                                 overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Process a list of products, creating chunks and embeddings
    
    Args:
        products: List of product dictionaries
        model: SentenceTransformer model for creating embeddings
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of processed documents with embeddings
    """

    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    all_documents = []
    
    print(f"Processing {len(products)} products with chunking...")
    
    for i, product in enumerate(products):
        if i % 100 == 0:
            print(f"Processed {i}/{len(products)} products")
        print(f"Processing product {i+1}/{len(products)}: {product.get('name', 'No Name')}")
        
        # Get chunks for this product
        chunks = chunker.chunk_product_description(product)

        if chunks:
            # Create embeddings for each chunk
            chunk_texts = [chunk['chunk_text'] for chunk in chunks]
            embeddings = model.encode(chunk_texts)
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                dictionary_chunk = {
                "id": chunk['product_id'] + "desc" + str(chunk['chunk_id']),
                "values":  embedding.tolist(),
                "metadata": {
                    "product_id": product.get('data_product'),
                    "type": "description",
                    "text": chunk['chunk_text'],
                    "name": product.get('name', ''),
                    "english_name": product.get('english_name'),
                    "categorys": product.get('categorys', ''),
                    "category_name": product.get('category_name', ''),
                    "brand": product.get('brand', ''),
                    "price": product.get('price'),
                    "total_rating": product.get('total_rating',''),
                    "average_rating": product.get('average_rating', ''),
                    "data_variant": product.get('data_variant', ''),
                    "item_count_by": product.get('item_count_by', '')
                }
                }

                all_documents.append(dictionary_chunk)

        

        # Embbedding the general information
        general_embedding, general_info = chunker.general_information_embedding(product, model)
        dictionary_chunk = {
            "id": product.get('data_product') + "general_info",
            "values": general_embedding,
            "metadata": {
                "product_id": product.get('data_product'),
                "type": "general_info",
                "text": general_info,
                "name": product.get('name', ''),
                "english_name": product.get('english_name'),
                "categorys": product.get('categorys', ''),
                "category_name": product.get('category_name', ''),
                "brand": product.get('brand', ''),
                "price": product.get('price'),
                "total_rating": product.get('total_rating',''),
                "average_rating": product.get('average_rating', ''),
                "data_variant": product.get('data_variant', ''),
                "item_count_by": product.get('item_count_by', '')
        }
        }
        
        all_documents.append(dictionary_chunk)

        # Comment embedding

        comments_embedding = chunker.comment_embedding(product, model)
        if comments_embedding is not True:
            for comment in comments_embedding:
                dictionary_chunk = {
                    "id": product.get('data_product') + "comment" + str(comment['comment_id']),
                    "values": comment['vector'],
                    "metadata": {
                        "product_id": product.get('data_product'),
                        "type": "comment",
                        "text": comment['comment'],
                        "name": product.get('name', ''),
                        "english_name": product.get('english_name'),
                        "categorys": product.get('categorys', ''),
                        "category_name": product.get('category_name', ''),
                        "brand": product.get('brand', ''),
                        "price": product.get('price'),
                        "total_rating": product.get('total_rating',''),
                        "average_rating": product.get('average_rating', ''),
                        "data_variant": product.get('data_variant', ''),
                        "item_count_by": product.get('item_count_by', '')
                        }
                }
                all_documents.append(dictionary_chunk)
        # Specification embedding
        spec_embedding = chunker.specificationinfo_embedding(product, model)
        if spec_embedding is not True:
            dictionary_chunk = {
                "id": product.get('data_product') + "specification",
                "values": spec_embedding,
                "metadata": {
                    "product_id": product.get('data_product'),
                    "type": "specification",
                    "text": product.get('specificationinfo', ''),
                    "name": product.get('name', ''),
                    "english_name": product.get('english_name'),
                    "categorys": product.get('categorys', ''),
                    "category_name": product.get('category_name', ''),
                    "brand": product.get('brand', ''),
                    "price": product.get('price'),
                    "total_rating": product.get('total_rating',''),
                    "average_rating": product.get('average_rating', ''),
                    "data_variant": product.get('data_variant', ''),
                    "item_count_by": product.get('item_count_by', '')
                }
            }
            all_documents.append(dictionary_chunk)
        # Ingredient info chunking
        ingredient_chunks = chunker.chunk_product_ingredientinfo(product)
        if ingredient_chunks:
            for ingredient_chunk in ingredient_chunks:
                dictionary_chunk = {
                    "id": ingredient_chunk['product_id'] + "ingredient" + str(ingredient_chunk['chunk_id']),
                    "values": model.encode([ingredient_chunk['chunk_text']]).tolist()[0],
                    "metadata": {
                        "product_id": ingredient_chunk['product_id'],
                        "type": "ingredient",
                        "text": ingredient_chunk['chunk_text'],
                        "name": product.get('name', ''),
                        "english_name": product.get('english_name'),
                        "categorys": product.get('categorys', ''),
                        "category_name": product.get('category_name', ''),
                        "brand": product.get('brand', ''),
                        "price": product.get('price'),
                        "total_rating": product.get('total_rating',''),
                        "average_rating": product.get('average_rating', ''),
                        "data_variant": product.get('data_variant', ''),
                        "item_count_by": product.get('item_count_by', '')
                    }
                }
                all_documents.append(dictionary_chunk)
        # Guide info chunking
        guide_chunks = chunker.chuck_product_guideinfo(product)
        if guide_chunks:
            for guide_chunk in guide_chunks:
                dictionary_chunk = {
                    "id": guide_chunk['product_id'] + "guide" + str(guide_chunk['chunk_id']),
                    "values": model.encode([guide_chunk['chunk_text']]).tolist()[0],
                    "metadata": {
                        "product_id": guide_chunk['product_id'],
                        "type": "guide",
                        "text": guide_chunk['chunk_text'],
                        "name": product.get('name', ''),
                        "english_name": product.get('english_name'),
                        "categorys": product.get('categorys', ''),
                        "category_name": product.get('category_name', ''),
                        "brand": product.get('brand', ''),
                        "price": product.get('price'),
                        "total_rating": product.get('total_rating',''),
                        "average_rating": product.get('average_rating', ''),
                        "data_variant": product.get('data_variant', ''),
                        "item_count_by": product.get('item_count_by', '')
                    }
                }
                all_documents.append(dictionary_chunk)
        print("length of all_documents", len(all_documents))
        if i == 0:
            break

    print(f"Created {len(all_documents)} document chunks from {len(products)} products")
    return all_documents